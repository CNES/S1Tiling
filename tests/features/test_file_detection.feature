#@tasks
Feature: Analysis of existing files
    Analyse existing files according to requested parameters like polarisation

    Scenario: Search VV-VH among VV-VH
        Given All S1 files are known
        When  VV-VH files are searched
        Then  VV files are found
        And   VH files are found
        And   No (other) files are found

    Scenario: Search VV among VV-VH
        Given All S1 files are known
        When  VV files are searched
        Then  VV files are found
        And   No (other) files are found

    Scenario: Search VH among VV-VH
        Given All S1 files are known
        When  VH files are searched
        Then  VH files are found
        And   No (other) files are found


    Scenario: Search VV-VH among VV
        Given All S1 VV files are known
        When  VV-VH files are searched
        Then  VV files are found
        And   No (other) files are found

    Scenario: Search VV among VV
        Given All S1 VV files are known
        When  VV files are searched
        Then  VV files are found
        And   No (other) files are found

    Scenario: Search VH among VV
        Given All S1 VV files are known
        When  VH files are searched
        Then  No (other) files are found


    Scenario: Search VV-VH among VH
        Given All S1 VH files are known
        When  VV-VH files are searched
        Then  VH files are found
        And   No (other) files are found

    Scenario: Search VH among VH
        Given All S1 VH files are known
        When  VH files are searched
        Then  VH files are found
        And   No (other) files are found

    Scenario: Search VV among VH
        Given All S1 VH files are known
        When  VV files are searched
        Then  No (other) files are found

