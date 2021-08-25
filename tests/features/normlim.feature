Feature: Norlim
    Existing S1 images shall be analysed to deduce normlim related tasks to
    execute.

    Background:
        Given A pipeline that computes LIA

    Scenario: Generate LIA tasks for a single S1 image
        Given a single S1 image

        When  dependencies are analysed
        And   tasks are generated

        Then  a LIA image is required
        And   LIA depends on XYZ image
        And   XYZ depends on DEM, DEMPROJ and BASE
        And   DEMPROJ depends on DEM and BASE
        And   DEM depends on BASE

        And   a LIA task is registered
        And   a XYZ task is registered
        And   a DEMPROJ task is registered
        And   a DEM task is registered
