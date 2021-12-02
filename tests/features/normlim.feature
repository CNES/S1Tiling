Feature: Norlim
    Existing S1 images shall be analysed to deduce normlim related tasks to
    execute.

    Scenario: Generate LIA tasks for a single S1 image
        Given A pipeline that computes LIA
        And   a single S1 image

        When  dependencies are analysed
        And   tasks are generated

        Then  a single LIA image is required
        And   LIA images depend on XYZ images
        And   XYZ images depend on DEM, DEMPROJ and BASE images
        And   DEMPROJ images depend on DEM and BASE images
        And   DEM images depend on BASE images

        And   LIA task(s) is(/are) registered
        And   XYZ task(s) is(/are) registered
        And   DEMPROJ task(s) is(/are) registered
        And   DEM task(s) is(/are) registered

    Scenario: Generate LIA tasks for a pair of VV+VH S1 images
        # Check a reduction of type 'any()': any one between vh or vv is good:
        # just keep one
        Given A pipeline that computes LIA
        And   a pair of VV + VH S1 images

        When  dependencies are analysed
        And   tasks are generated

        Then  a single LIA image is required
        And   LIA images depend on XYZ images
        And   XYZ images depend on DEM, DEMPROJ and BASE images
        And   DEMPROJ images depend on DEM and BASE images
        And   DEM images depend on BASE images

        And   LIA task(s) is(/are) registered
        And   XYZ task(s) is(/are) registered
        And   DEMPROJ task(s) is(/are) registered
        And   DEM task(s) is(/are) registered

    Scenario: Generate LIA tasks for a series of S1 VV images
        # Check a single LIA task will be registered even w/ multiple input
        # images of different acquisition date. => Keep only one LIA
        Given A pipeline that fully computes in LIA S2 geometry
        And   a series of S1 VV images

        When  dependencies are analysed
        And   tasks are generated

        Then  a single S2 LIA image is required
        # TODO fix the dependencies
        And   concat LIA depends on 2 ortho LIA images
        And   2 ortho LIA images depend on two LIA images
        And   LIA images depend on XYZ images
        And   XYZ images depend on DEM, DEMPROJ and BASE images
        And   DEMPROJ images depend on DEM and BASE images
        And   DEM images depend on BASE images

        And   a concat LIA task is registered
        And   ortho LIA task(s) is(/are) registered
        And   LIA task(s) is(/are) registered
        And   XYZ task(s) is(/are) registered
        And   DEMPROJ task(s) is(/are) registered
        And   DEM task(s) is(/are) registered
