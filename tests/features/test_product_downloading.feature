Feature: Test download requests
    Test download requests given requirements and detected files

    Scenario: Everything was downloaded and generated
        Given Request on 8th jan
        #Given Request on all dates
        And   All S1 files are known
        And   All S2 files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  None are requested for download

    Scenario: Everything was downloaded and nothing was generated
        Given Request on 8th jan
        #Given Request on all dates
        And   All S1 files are known
        And   No S2 files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  None are requested for download

    Scenario: Nothing was downloaded and everything was generated
        Given Request on 8th jan
        #Given Request on all dates
        And   No S1 files are known
        And   All S2 files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  None are requested for download

    Scenario: Nothing was downloaded and nothing was generated
        Given Request on 8th jan
        #Given Request on all dates
        And   No S1 files are known
        And   No S2 files are known
        And   All products are available for download
        When  Searching which S1 files to download
        Then  All are requested for download

# + scenarios with VV / VH mismatchs
# + fname_fmt

