#@tasks
Feature: Dependencies and Tasks
    Existing products shall be analysed
    to deduce tasks to execute

    Examples:
        |builds        | a  |
        |doesn't build | no |
        |builds        | a  |

    Background:
        Given A pipeline that calibrates and orthorectifies
        And   that concatenates

    #@txxxxxx_mask
    Scenario Outline: Orthorectify and concatenate two S1 images
        Given that <builds> masks
        And   two S1 images

        When  dependencies are analysed
        And   tasks are generated

        Then  a txxxxxx S2 file is required, and <a> mask is required
        And   it depends on 2 ortho files (and two S1 inputs), and <a> mask on a concatenated product
        And   a concatenation task is registered and produces txxxxxxx S2 file and <a> mask
        And   two orthorectification tasks are registered

    Scenario Outline: Orthorectify and concatenate a single S1 image
        Given that <builds> masks
        And   a single S1 image

        When  dependencies are analysed
        And   tasks are generated

        Then  a t-chrono S2 file is required, and <a> mask is required
        And   it depends on one ortho file (and one S1 input), and <a> mask on a concatenated product
        And   a concatenation task is registered and produces t-chrono S2 file, and <a> mask
        And   a single orthorectification task is registered
        But   dont orthorectify the second product

    Scenario Outline: Orthorectify a single S1 image and concatenate it to a tmp FullOrtho
        Given that <builds> masks
        And   a single S1 image
        And   a FullOrtho tmp image

        When  dependencies are analysed
        And   tasks are generated

        Then  a txxxxxx S2 file is required, and <a> mask is required
        And   it depends on 2 ortho files (and two S1 inputs), and <a> mask on a concatenated product
        And   a concatenation task is registered and produces txxxxxxx S2 file and <a> mask
        And   a single orthorectification task is registered
        And   it depends on the existing FullOrtho tmp product

    Scenario Outline: concatenate two tmp FullOrtho
        Given that <builds> masks
        And   two FullOrtho tmp images

        When  dependencies are analysed
        And   tasks are generated

        Then  a txxxxxx S2 file is required, and <a> mask is required
        And   it depends on 2 ortho files (and two S1 inputs), and <a> mask on a concatenated product
        And   a concatenation task is registered and produces txxxxxxx S2 file and <a> mask
        And   no orthorectification tasks is registered
        And   it depends on two existing FullOrtho tmp products

    Scenario Outline: concatenate a single tmp FullOrtho
        Given that <builds> masks
        And    a FullOrtho tmp image

        When  dependencies are analysed
        And   tasks are generated

        Then  a t-chrono S2 file is required, and <a> mask is required
        And   it depends on second ortho file (and second S1 input), and <a> mask on a concatenated product
        And   a concatenation task is registered and produces t-chrono S2 file, and <a> mask
        And   no orthorectification tasks is registered
        And   it depends on the existing FullOrtho tmp product

    # Other alternate scenarios:
    # x2 for masks
