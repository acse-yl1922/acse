"""Test Module."""

from flood_tool import tool, geo
import numpy as np
from pytest import mark


tool = tool.Tool()


def test_get_easting_northing():
    """Check """

    data = tool.get_easting_northing(['BN1 5PF'])

    assert np.isclose(data.iloc[0].easting, 530401.0)
    assert np.isclose(data.iloc[0].northing, 105619.0)

@mark.xfail  # We expect this test to fail until we write some code for it.
def test_get_lat_long():
    """Check """

    data = tool.get_lat_long(['BN1 5PF'])

    assert np.isclose(data.iloc[0].latitude, 50.8354, 1.0e-2)
    assert np.isclose(data.iloc[0].longitude, -0.1495, 1.0e-2)

def test_get_all_features_for_postcodes():
    """Check """
    
    data = tool.get_all_features_for_postcodes(['BN1 5PF'])
    
    assert data.iloc[0].localAuthority == "Brighton and Hove"
    assert data.iloc[0].soilType == "Unsurveyed/Urban"

def test_get_easting_northing_from_gps_lat_long():
    
    easting, northing = geo.get_easting_northing_from_gps_lat_long([50.8354], [-0.1495])
    
    assert np.isclose(easting, 530401.0, 1.0e2)
    assert np.isclose(northing, 105619.0, 1.0e2)

def test_get_gps_lat_long_from_easting_northing():
    
    lat, long = geo.get_gps_lat_long_from_easting_northing([530401.0], [105619.0])
    
    assert np.isclose(lat, 50.8354, 1.0e-2)
    assert np.isclose(long, -0.1495, 1.0e-2)
    

if __name__ == "__main__":
    test_get_easting_northing()
    test_get_lat_long()
    test_get_all_features_for_postcodes()
    test_get_easting_northing_from_gps_lat_long()
    test_get_gps_lat_long_from_easting_northing()

