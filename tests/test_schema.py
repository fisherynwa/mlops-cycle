import pytest
from pydantic import ValidationError
 
from src.serve import ClaimFeatures
 
 
class TestClaimFeatures:
    """Tests for the ClaimFeatures pydantic model: validation of input fields."""
    def test_valid_input_passes(self):
        c = ClaimFeatures(age=45, bmi=30.5, smoker="yes")
        assert c.age == 45
        assert c.bmi == 30.5
        assert c.smoker == "yes"
 
    def test_age_too_high_rejected(self):
        with pytest.raises(ValidationError):
            ClaimFeatures(age=200, bmi=30.5, smoker="yes")   
 
    def test_negative_age_rejected(self):
        with pytest.raises(ValidationError):
            ClaimFeatures(age=-5, bmi=30.5, smoker="yes")    
 
    def test_bmi_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            ClaimFeatures(age=45, bmi=5.0, smoker="yes")    
 
    def test_invalid_smoker_value_rejected(self):
        with pytest.raises(ValidationError):
            ClaimFeatures(age=45, bmi=30.5, smoker="maybe")  
 
    def test_missing_field_rejected(self):
        with pytest.raises(ValidationError):
            ClaimFeatures(age=45, bmi=30.5)                 
