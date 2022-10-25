Feature: Test behaviour on non-available products

    Test available S1 product pairs when products could not be downloaded

    Scenario: Everything is available
        Given S1 product 0 has been downloaded
        And   S1 product 1 has been downloaded
        When  Filtering products to use
        Then  All S2 products will be generated

    Scenario: One product is offline
        Given S1 product 0 has been downloaded
        And   S1 product 1 download has timed-out
        When  Filtering products to use
        Then  No S2 product will be generated

    Scenario: Two paired products are offline
        Given S1 product 0 download has timed-out
        And   S1 product 1 download has timed-out
        When  Filtering products to use
        Then  0 S2 product(s) will be generated

    Scenario: Some are available, some are not... 1/3
        Given Request on all dates
        And   S1 product 0 has been downloaded
        And   S1 product 1 has been downloaded
        And   S1 product 2 has been downloaded
        And   S1 product 3 download has timed-out
        And   S1 product 4 download has timed-out
        And   S1 product 5 download has timed-out
        When  Filtering products to use
        Then  1 S2 product(s) will be generated
        And   S2 product n° 0 will be generated

    Scenario: Some are available, some are not... 2/3
        Given Request on all dates
        And   S1 product 0 has been downloaded
        And   S1 product 1 download has timed-out
        And   S1 product 2 has been downloaded
        And   S1 product 3 has been downloaded
        And   S1 product 4 download has timed-out
        And   S1 product 5 download has timed-out
        When  Filtering products to use
        Then  1 S2 product(s) will be generated
        And   S2 product n° 1 will be generated

    Scenario: Some are available, some are not... 3/3
        Given Request on all dates
        And   S1 product 0 download has timed-out
        And   S1 product 1 download has timed-out
        And   S1 product 2 download has timed-out
        And   S1 product 3 has been downloaded
        And   S1 product 4 has been downloaded
        And   S1 product 5 has been downloaded
        When  Filtering products to use
        Then  1 S2 product(s) will be generated
        And   S2 product n° 2 will be generated


