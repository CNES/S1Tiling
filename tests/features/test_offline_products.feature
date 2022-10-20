Feature: Test behaviour on non-available products

    Test available S1 product pairs when products could not be downloaded

    Scenario: Everything is available
        Given All S1 products have been downloaded
        When  Filtering products to use
        Then  All S2 products will be generated

    Scenario: One product is offline
        Given First S1 product has been downloaded
        And   Second S1 product download timed-out
        When  Filtering products to use
        Then  No S2 product will be generated

    Scenario: Two paired products are offline
        Given First S1 product download timed-out
        And   Second S1 product download timed-out
        When  Filtering products to use
        Then  No S2 product will be generated

