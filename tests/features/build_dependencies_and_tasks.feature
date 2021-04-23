#@tasks
Feature: Dependencies and Tasks
    Existing products shall be analysed
    to deduce tasks to execute

    Scenario: Orthorectify and concatenate two S1 images
        Given A pipeline that calibrates and orthorectifies
        And   that concatenates
        And   two S1 images

        When  dependencies are analysed
        And   tasks are generated

        Then  a txxxxxx S2 file is required
        And   it depends on 2 ortho files (and two S1 inputs)
        And   a concatenation task is registered and produces txxxxxxx S2 file
        And   two orthorectification tasks are registered

    @chrono
    Scenario: Orthorectify and concatenate a single S1 image
        Given A pipeline that calibrates and orthorectifies
        And   that concatenates
        And   a single S1 image

        When  dependencies are analysed
        And   tasks are generated

        Then  a t-chrono S2 file is required
        And   it depends on one ortho file (and one S1 input)
        And   a concatenation task is registered and produces t-chrono S2 file
        And   a single orthorectification tasks is registered
        But   dont orthorectify the second product

    Scenario: Orthorectify a single S1 image and concatenate it to a tmp FullOrtho
        Given A pipeline that calibrates and orthorectifies
        And   that concatenates
        And   a single S1 image
        And   a FullOrtho tmp image

        When  dependencies are analysed
        And   tasks are generated

        Then  a txxxxxx S2 file is required
        And   it depends on 2 ortho files (and two S1 inputs)
        And   a concatenation task is registered and produces txxxxxxx S2 file
        And   a single orthorectification tasks is registered
        And   it depends on the existing FullOrtho tmp product

    Scenario: concatenate two tmp FullOrtho
        Given A pipeline that calibrates and orthorectifies
        And   that concatenates
        And   two FullOrtho tmp images

        When  dependencies are analysed
        And   tasks are generated

        Then  a txxxxxx S2 file is required
        And   it depends on 2 ortho files (and two S1 inputs)
        And   a concatenation task is registered and produces txxxxxxx S2 file
        And   no orthorectification tasks is registered
        And   it depends on two existing FullOrtho tmp products

    Scenario: concatenate a single tmp FullOrtho
        Given A pipeline that calibrates and orthorectifies
        And   that concatenates
        And   a FullOrtho tmp image

        When  dependencies are analysed
        And   tasks are generated

        Then  a t-chrono S2 file is required
        And   it depends on second ortho file (and second S1 input)
        And   a concatenation task is registered and produces t-chrono S2 file
        And   no orthorectification tasks is registered
        And   it depends on the existing FullOrtho tmp product

    # Other alternate scenarios:
    # x2 for masks
