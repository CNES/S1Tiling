Feature: Test download requests
    Test download requests given requirements and detected files

        Examples:
            | dates     |
            | 8th jan   |
            | all dates |

    Scenario Outline: Everything was downloaded and generated
        Given Request on <dates>
        And   All S1 files are known
        And   All S2 files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  None are requested for download

    Scenario Outline: Everything was downloaded and nothing was generated
        Given Request on <dates>
        And   All S1 files are known
        And   No S2 files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  None are requested for download

    Scenario Outline: Nothing was downloaded and everything was generated
        Given Request on <dates>
        And   No S1 files are known
        And   All S2 files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  None are requested for download

    Scenario Outline: Nothing was downloaded and nothing was generated
        Given Request on <dates>
        And   No S1 files are known
        And   No S2 files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  All are requested for download

    # + scenarios with VV / VH mismatchs
    Scenario Outline: Everything was downloaded and all VV were generated and requested
        Given Request on <dates>
        And   Request on VV
        And   All S1 files are known
        And   All S2 VV files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  None are requested for download

    Scenario Outline: Everything was downloaded and all VV were generated but VH requested
        Given Request on <dates>
        And   Request on VH
        And   All S1 files are known
        And   All S2 VV files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  None are requested for download

    Scenario Outline: Nothing was downloaded and all VV was generated and requested
        Given Request on <dates>
        And   Request on VV
        And   No S1 files are known
        And   All S2 VV files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  None are requested for download

    Scenario Outline: Nothing was downloaded and all VV was generated but VH requested
        Given Request on <dates>
        And   Request on VH
        And   No S1 files are known
        And   All S2 VV files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  All are requested for download


    # + scenarios with fname_fmt mismatch
    Scenario Outline: Nothing was downloaded and everything was generated but for another calibration
        Given Request on <dates>
        And   Request for _beta
        And   No S1 files are known
        And   All S2 files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  All are requested for download

    Scenario Outline: Nothing was downloaded and everything was generated but for another fname_fmt
        Given Request on <dates>
        And   Request with default fname_fmt_concatenation
        And   No S1 files are known
        And   All S2 files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  All are requested for download


