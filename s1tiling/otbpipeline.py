#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright (c) CESBIO. All rights reserved.
#
#   See LICENSE for details.
#
#   This software is distributed WITHOUT ANY WARRANTY; without even
#   the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#   PURPOSE.  See the above copyright notices for more information.
#
# =========================================================================
#
# Authors: Thierry KOLECK (CNES)
#          Luc HERMITTE (CS Group)
# =========================================================================

"""
This module provides pipeline for chaining OTB applications, and a pool to execute them.
"""

import os
from abc import ABC, abstractmethod
import logging
import logging.handlers
import multiprocessing
import otbApplication as otb
import s1tiling.Utils as Utils


def worker_config(q):
    """
    Worker configuration function called by Pool().

    It takes care of initializing the queue handler in the subprocess.

    Params:
        :q: multiprocessing.Queue used for passing logging messages from worker to main process.
    """
    qh = logging.handlers.QueueHandler(q)
    logger = logging.getLogger()
    logger.addHandler(qh)


def in_filename(meta):
    """
    Helper accessor to access the input filename of a `Step`.
    """
    assert('in_filename' in meta)
    return meta['in_filename']
    # return meta['in_filename'] if 'in_filename' in meta else meta['basename']

def out_filename(meta):
    """
    Helper accessor to access the ouput filename of a `Step`.
    """
    return meta.get('out_filename')


class AbstractStep(object):
    """
    Internal root class for all actual `Step`s.

    There are three kinds of steps:
    - `FirstStep` that contains information about input files
    - `Step` that registers an otbapplication binding
    - `StoreStep` that momentarilly disconnect on-memory pipeline to force storing of the resulting file.

    The step will contain information like the current input file, the current output file...
    """
    def __init__(self, *argv, **kwargs):
        """
        constructor
        """
        meta = kwargs
        assert('basename' in meta)
        # Clear basename from any noise
        self._meta   = meta

    @property
    def is_first_step(self):
        return True

    @property
    def meta(self):
        return self._meta

    @property
    def basename(self):
        """
        basename property will be used to generate all future output filenames.
        """
        return self._meta['basename']

    @property
    def out_filename(self):
        """
        Property that returns the name of the file produced by the current step.
        """
        assert('out_filename' in self._meta)
        return self._meta['out_filename']

    @property
    def shall_store(self):
        """
        No step required its result to be stored on disk and to break in_memory
        connection by default.
        However, the artificial Step produced by `Store` factory will force the
        result of the _previous_ app to be stored on disk.
        """
        return False

    def release_app(self):
        """
        Makes sure that steps with applications are releasing the application
        """
        pass


class _StepWithOTBApplication(AbstractStep):
    """
    Internal intermediary type for `Step` that have an application object.
    Not meant to be used directly.

    Parent type for:
    - `Step`  that will own the application
    - and `StoreStep` that will just reference the application from the previous step
    """
    def __init__(self, app, *argv, **kwargs):
        """
        constructor
        """
        # logging.debug("Create Step(%s, %s)", app, meta)
        super().__init__(*argv, **kwargs)
        self._app    = app

    def __del__(self):
        """
        Makes sure the otb app is released
        """
        if self._app:
            self.release_app()

    def release_app(self):
        # Only `Step` will delete, here we just reset the reference
        self._app = None

    @property
    def app(self):
        return self._app

    @property
    def is_first_step(self):
        return self._app is None

    @property
    def param_out(self):
        return self._out


class Step(_StepWithOTBApplication):
    """
    Interal specialized `Step` that holds a binding to an OTB Application.

    The application binding is expected to be built by a dedicated `StepFactory` and passed to the constructor.
    """
    def __init__(self, app, *argv, **kwargs):
        """
        constructor
        """
        # logging.debug("Create Step(%s, %s)", app, meta)
        super().__init__(app, *argv, **kwargs)
        self._out    = kwargs.get('param_out', 'out')

    def release_app(self):
        del(self._app)
        super().release_app() # resets self._app to None

    def ExecuteAndWriteOutput(self):
        """
        Method to call on the last step of a pipeline.
        """
        assert(self._app)
        if not self.meta.get('dryrun', False):
            self._app.ExecuteAndWriteOutput()
        if 'post' in self.meta:
            for hook in meta['post']:
                hook(self.meta)


class StepFactory(ABC):
    """
    Abstract factory for `Step`.

    Meant to be inherited for each possible OTB application used in a pipeline.
    """
    def __init__(self, appname, *argv, **kwargs):
        self._in         = kwargs.get('param_in',  'in')
        self._out        = kwargs.get('param_out', 'out')
        self._appname    = appname
        logging.debug("new StepFactory(%s)", appname)

    @property
    def appname(self):
        return self._appname

    @property
    def param_in(self):
        return self._in

    @property
    def param_out(self):
        return self._out

    @abstractmethod
    def parameters(self, meta):
        pass

    @abstractmethod
    def build_step_output_filename(self, meta):
        pass

    @abstractmethod
    def output_directory(self, meta):
        pass

    def set_output_pixel_type(self, app, meta):
        """
        Permits to have steps force the output pixel data.
        Does nothing by default.
        Override this method to change the output pixel type.
        """
        pass

    def complete_meta(self, meta): # to be overridden
        meta = meta.copy()
        meta['in_filename']  = out_filename(meta)
        meta['out_filename'] = self.build_step_output_filename(meta)
        return meta

    def create_step(self, input: Step, in_memory: bool):
        # TODO: handle filename transformations
        # TODO: distinguish step description & step
        meta = self.complete_meta(input.meta)
        if self.appname:
            app = otb.Registry.CreateApplication(self.appname)
            if not app:
                raise RuntimeError("Cannot create OTB application '"+self.appname+"'")
            parameters = self.parameters(meta)
            if input.is_first_step:
                parameters[self.param_in] = input.out_filename
            else:
                app.ConnectImage(self.param_in, input.app, input.param_out)
                this_step_is_in_memory = in_memory and not input.shall_store
                logging.debug("Chaining %s in memory: %s", self.appname, this_step_is_in_memory)
                app.PropagateConnectMode(this_step_is_in_memory)
                # When this is not a first step, we need to clear the input parameters
                # from its list, otherwise some OTB applications may comply
                del parameters[self.param_in]

            self.set_output_pixel_type(app, meta)
            logging.debug('Params: %s: %s', self.appname, parameters)
            app.SetParameters(parameters) # ordre à vérifier!
            meta['param_out'] = self.param_out
            return Step(app, **meta)
        else:
            # Return previous app?
            return AbstractStep(**meta)



class Pipeline(object):
    """
    Pipeline of OTB applications.

    It's instanciated as a list of `AbstractStep`s.
    `Step.ExecuteAndWriteOutput()` will be executed on the last step of the
    pipeline.

    Internal class only meant to be used by  `Pool`.
    """
    ## Should we inherit from contextlib.ExitStack?
    def __init__(self, do_measure, in_memory):
        self.__pipeline = []
        self.__do_measure = do_measure
        self.__in_memory  = in_memory
    #def __enter__(self):
    #    return self
    #def __exit__(self, type, value, traceback):
    #    for crt in self.__pipeline:
    #        crt.release_app() # Make sure to release application memory
    #    return False

    def set_input(self, input):
        self.__input = input
    def push(self, otbstep):
        self.__pipeline += [otbstep]
    def do_execute(self):
        # print("LOG:", os.environ['OTB_LOGGER_LEVEL'])
        assert(self.__pipeline) # shall not be empty!
        app_names = [str(self.__input.out_filename)]
        steps     = [self.__input]
        for crt in self.__pipeline:
            step = crt.create_step(steps[-1], self.__in_memory)
            steps += [step]
            app_names += [crt.appname]

        pipeline_name = '|'.join(app_names)
        # TODO: if last output file already exists: permit to ignore...
        with Utils.ExecutionTimer('-> '+pipeline_name, self.__do_measure) as t:
            assert(steps)
            # TODO: catch execute failure, and report it!
            steps[-1].ExecuteAndWriteOutput()
        for step in steps:
            step.release_app() # Make sure to release application memory
        return pipeline_name + ' > ' + steps[-1].out_filename


# TODO: try to make it static...
def execute1(pipeline):
    return pipeline.do_execute()


class PoolOfOTBExecutions(object):
    """
    Internal multiprocess Pool of OTB pipelines.
    """
    def __init__(self, title, do_measure, nb_procs, nb_threads, log_queue, log_queue_listener):
        """
        constructor
        """
        self.__pool = []
        self.__title               = title
        self.__do_measure          = do_measure
        self.__nb_procs            = nb_procs
        self.__nb_threads          = nb_threads
        self.__log_queue           = log_queue
        self.__log_queue_listener  = log_queue_listener

    def new_pipeline(self, **kwargs):
        in_memory = kwargs.get('in_memory', True)
        pipeline = Pipeline(self.__do_measure, in_memory)
        self.__pool += [pipeline]
        return pipeline

    def process(self):
        nb_cmd = len(self.__pool)

        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(self.__nb_threads)
        os.environ['OTB_LOGGER_LEVEL'] = 'DEBUG'
        if False: # debug OTB applications with gdb => do not spawn process!
            execute1(self.__pool[0])
        else:
            with multiprocessing.Pool(self.__nb_procs, worker_config, [self.__log_queue]) as pool:
                self.__log_queue_listener.start()
                for count, result in enumerate(pool.imap_unordered(execute1, self.__pool), 1):
                    logging.info("%s correctly finished", result)
                    logging.info(' --> %s... %s%%', self.__title, count*100./nb_cmd)

                pool.close()
                pool.join()
                self.__log_queue_listener.stop()


class Processing(object):
    """
    Entry point for executing multiple instance of the same pipeline of
    different inputs.

    1. The object is initialized with a log queue and its listener
    2. The pipeline is registered with a list of `StepFactory`s
    3. The processing is done on a list of `FirstStep`s
    """
    def __init__(self, cfg):
        self.__log_queue          = cfg.log_queue
        self.__log_queue_listener = cfg.log_queue_listener
        self.__cfg                = cfg
        # self.__factory_steps      = []

    def register_pipeline(self, factory_steps):
        self.__factory_steps = factory_steps

    def process(self, startpoints):
        # TODO: forward the exact config params: thr, process
        pool = PoolOfOTBExecutions("testpool", True, 2, 1,
            self.__log_queue, self.__log_queue_listener)
        for startpoint in startpoints:
            logging.info("register processing of %s", startpoint.basename)
            pipeline = pool.new_pipeline(in_memory=True)
            pipeline.set_input(startpoint)
            for factory in self.__factory_steps:
                pipeline.push(factory(self.__cfg))

        logging.debug('Launch pipelines')
        pool.process()


# ======================================================================
class FirstStep(AbstractStep):
    """
    First Step:
    - no application executed
    """
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        if not 'out_filename' in self._meta:
            # If not set through the parameters, set it from the basename + out dir
            self._meta['out_filename'] = self._meta['basename']
        working_directory, basename = os.path.split(self._meta['basename'])
        self._meta['basename'] = basename


class StoreStep(_StepWithOTBApplication):
    def __init__(self, previous: Step):
        assert(not previous.is_first_step)
        super().__init__(previous._app, *[], **previous.meta)
        self._out    = previous.param_out

    @property
    def shall_store(self):
        return True

    def ExecuteAndWriteOutput(self):
        raise TypeError("A StoreStep is not meant to be the last step of a pipeline!!!")


class Store(StepFactory):
    """
    Artificial Step that forces the result of the previous app to be stored on
    disk by breaking in-memory connection.
    """
    def __init__(self, appname, *argv, **kwargs):
        super().__init__("(StoreOnFile)", *argv, **kwargs)
    def create_step(self, input: Step, in_memory: bool):
        return StoreStep(input)

    # abstract methods...
    def parameters(self, meta):
        raise TypeError("No way to ask for the parameters from a StoreFactory")
    def output_directory(self, meta):
        raise TypeError("No way to ask for output dir of a StoreFactory")
    def build_step_output_filename(self, meta):
        raise TypeError("No way to ask for the output filename of a StoreFactory")



# ======================================================================
if __name__ == '__main__':
    import sys
    from s1tiling.configuration import Configuration
    from s1tiling.otbwrappers import AnalyseBorders, Calibrate, CutBorders, OrthoRectify, Concatenate

    CFG = sys.argv[1]
    cfg=Configuration(CFG)
    # cfg.tmp_srtm_dir       = tempfile.mkdtemp(dir=cfg.tmpdir)
    cfg.tmp_srtm_dir       = 'tmp_srtm_dir'

    # The pool of jobs
    tile_name = '33NWB'
    tile_origin = [(14.9998201759, 1.8098185887), (15.9870050338, 1.8095484335), (15.9866155411, 0.8163071941), (14.9998202469, 0.8164290331000001)]
    safename = 'data_raw/S1A_IW_GRDH_1SDV_20200108T044215_20200108T044240_030704_038506_D953.SAFE'
    filename = safename+'/measurement/s1a-iw-grd-vv-20200108t044215-20200108t044240-030704-038506-001.tiff'
    pre_ortho_filename = safename+'/measurement/s1a-iw-grd-vv-20200108t044215-20200108t044240-030704-038506-001_OrthoReady.tiff'
    manifest = safename+'/manifest.safe'

    # os.symlink(os.path.join(cfg.srtm,srtm_tile),os.path.join(cfg.tmp_srtm_dir,srtm_tile))


    process = Processing(cfg)
    # process.register_pipeline([Calibrate, OrthoRectify])
    # process.register_pipeline([AnalyseBorders, Calibrate, CutBorders])
    # process.register_pipeline([AnalyseBorders, CutBorders])
    process.register_pipeline([AnalyseBorders, Calibrate, CutBorders, OrthoRectify])
    startpoint = FirstStep(tile_name=tile_name, tile_origin=tile_origin,
            manifest=manifest, basename=filename)
    process.process([startpoint])

    # process.register_pipeline([AnalyseBorders, OrthoRectify])
    # process.process(tile_name, tile_origin, manifest, pre_ortho_filename)

