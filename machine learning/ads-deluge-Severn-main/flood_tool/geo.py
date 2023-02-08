from numpy import array, asarray, mod, sin, cos, tan, sqrt, arctan2, floor, rad2deg, deg2rad, stack, exp, absolute, arccos
from scipy.linalg import inv

__all__ = ['get_easting_northing_from_gps_lat_long',
           'get_gps_lat_long_from_easting_northing']

class Ellipsoid(object):
    """ Data structure for a global ellipsoid. """

    def __init__(self, a, b, F_0):
        self.a = a
        self.b = b
        self.n = (a-b)/(a+b)
        self.e2 = (a**2-b**2)/a**2
        self.F_0 = F_0
        self.H=0

class Datum(Ellipsoid):
    """ Data structure for a global datum. """

    def __init__(self, a, b, F_0, phi_0, lam_0, E_0, N_0, H):
        super().__init__(a, b, F_0)
        self.phi_0 = phi_0
        self.lam_0 = lam_0
        self.E_0 = E_0
        self.N_0 = N_0
        self.H = H

def dms2rad(deg, min=0, sec=0):
    """Convert degrees, minutes, seconds to radians.
    
    Parameters
    ----------
    deg: array_like
        Angle in degrees.
    min: array_like
        (optional) Angle component in minutes.
    sec: array_like
        (optional) Angle component in seconds.

    Returns
    -------
    numpy.ndarray
        Angle in radians.
    """
    deg = asarray(deg)
    return deg2rad(deg+min/60.+sec/3600.)

def rad2dms(rad, dms=False):
    """Convert radians to degrees, minutes, seconds.

    Parameters
    ----------

    rad: array_like
        Angle in radians.
    dms: bool
        Use degrees, minutes, seconds format. If False, use decimal degrees.

    Returns
    -------
    numpy.ndarray
        Angle in degrees, minutes, seconds or decimal degrees.
    """

    rad = asarray(rad)
    deg = rad2deg(rad)
    if dms:
        min = 60.0*mod(deg, 1.0)
        sec = 60.0*mod(min, 1.0)
        return stack((floor(deg), floor(min), sec.round(4)))
    else:
        return deg

osgb36 = Datum(a=6377563.396,
               b=6356256.910,
               F_0=0.9996012717,
               phi_0=deg2rad(49.0),
               lam_0=deg2rad(-2.),
               E_0=400000,
               N_0=-100000,
               H=24.7)

wgs84 = Ellipsoid(a=6378137, 
                  b=6356752.3142,
                  F_0=0.9996)

def lat_long_to_xyz(phi, lam, rads=False, datum=osgb36):
    """Convert input latitude/longitude in a given datum into
    Cartesian (x, y, z) coordinates.

    Parameters
    ----------

    phi: array_like
        Latitude in degrees (if radians=False) or radians (if radians=True).
    lam: array_like
        Longitude in degrees (if radians=False) or radians (if radians=True).
    rads: bool (optional)
        If True, input latitudes and longitudes are in radians.
    datum: Datum (optional)
        Datum to use for conversion.
    """
    if not rads:
        phi = deg2rad(phi)
        lam = deg2rad(lam)

    nu = datum.a*datum.F_0/sqrt(1-datum.e2*sin(phi)**2)
  
    return array(((nu+datum.H)*cos(phi)*cos(lam),
                  (nu+datum.H)*cos(phi)*sin(lam),
                  ((1-datum.e2)*nu+datum.H)*sin(phi)))

def xyz_to_lat_long(x,y,z, rads=False, datum=osgb36):

    p = sqrt(x**2+y**2)

    lam = arctan2(y, x)
    phi = arctan2(z,p*(1-datum.e2))

    for _ in range(10):

        nu = datum.a*datum.F_0/sqrt(1-datum.e2*sin(phi)**2)
        dnu = -datum.a*datum.F_0*cos(phi)*sin(phi)/(1-datum.e2*sin(phi)**2)**1.5

        f0 = (z + datum.e2*nu*sin(phi))/p - tan(phi)
        f1 = datum.e2*(nu**cos(phi)+dnu*sin(phi))/p - 1.0/cos(phi)**2
        phi -= f0/f1

    if not rads:
        phi = rad2dms(phi)
        lam = rad2dms(lam)

    return phi, lam

def get_M(phi, phi_0, n):
    phi_diff = phi - phi_0
    phi_sum = phi + phi_0
    
    x1 = (1 + n + ((5/4) * (n ** 2)) + ((5/4) * (n ** 3))) * phi_diff

    x2 = ((3 * n) + (3 * (n ** 2)) + ((21/8) * (n ** 3))) * sin(phi_diff) * cos(phi_sum)

    x3 = (( (15/8) * (n ** 2)) + ((15/8) * (n ** 3))) * sin(2 * (phi_diff)) * cos(2 * (phi_sum))

    x4 = (35 / 24) * (n ** 3) * sin(3 * phi_diff) * cos(3 * (phi_sum))

    M = wgs84.b * wgs84.F_0 * (x1 - x2 + x3 - x4)

    return M

def get_easting_northing_from_gps_lat_long(phi, lam, rads=False):
    """ Get OSGB36 easting/northing from GPS latitude and longitude pairs.

    Parameters
    ----------
    phi: float/arraylike
        GPS (i.e. WGS84 datum) latitude value(s)
    lam: float/arrayling
        GPS (i.e. WGS84 datum) longitude value(s).
    rads: bool (optional)
        If true, specifies input is is radians.
    Returns
    -------
    numpy.ndarray
        Easting values (in m)
    numpy.ndarray
        Northing values (in m)
        Examples
    --------
    >>> get_easting_northing_from_gps_lat_long([55.5], [-1.54])
    (array([429157.0]), array([623009]))
    References
    ----------
    Based on the formulas in "A guide to coordinate systems in Great Britain".
    See also https://webapps.bgs.ac.uk/data/webservices/convertForm.cfm

    conversion formula from: https://www.ordnancesurvey.co.uk/documents/resources/guide-coordinate-systems-great-britain.pdf
    """
    eastings = []
    northings = []
    phi = array(phi)
    lam = array(lam)

    if not rads:
        phi = deg2rad(phi)
        lam = deg2rad(lam)

    for i in range(len(phi)):

        n = wgs84.n

        v = wgs84.a * wgs84.F_0 * (1 - (wgs84.e2 * (sin(phi[i]) ** 2))) ** -0.5

        rho = wgs84.a * wgs84.F_0 * (1 - wgs84.e2) * (1 - (wgs84.e2 * sin(phi[i]) **2)) ** -1.5

        eta2 = (v / rho) - 1

        M = get_M(phi[i], osgb36.phi_0, n)

        I = M + osgb36.N_0

        II = (v / 2) * sin(phi[i]) * cos(phi[i])

        III = (v / 24) * sin(phi[i]) * (cos(phi[i])**3) * (5 - (tan(phi[i]) ** 2) + (9 * eta2))

        IIIA = (v / 720) * sin(phi[i]) * (cos(phi[i]) ** 5) * (61 - (58 * (tan(phi[i]) ** 2)) + tan(phi[i])**4)

        IV = v * cos(phi[i])

        V = (v / 6) * (cos(phi[i]) ** 3) * ((v / rho) - (tan(phi[i]) ** 2))

        VI = (v / 120) * (cos(phi[i]) ** 5) * (5 - (18 * (tan(phi[i]) ** 2)) + (tan(phi[i]) ** 4) + (14 * eta2) - (58 * (tan(phi[i]) ** 2) * eta2))

        lam_diff = lam[i] - osgb36.lam_0

        N = I + (II * (lam_diff ** 2)) + (III * (lam_diff ** 4)) + (IIIA * (lam_diff ** 6))

        E = osgb36.E_0 + (IV * lam_diff) + (V * lam_diff ** 3) + (VI * lam_diff ** 5)

        eastings.append(E)
        northings.append(N)

    return array(eastings), array(northings)


def get_gps_lat_long_from_easting_northing(east, north, rads=False, dms=False):
    """ Get GPS latitude and longitude from OSGB36 easting/northing pairs.
    Parameters
    ----------
    east: float/arraylike
        OSGB36 easting value(s) (in m).
    north: float/arrayling
        OSGB36 easting value(s) (in m).
    rads: bool (optional)
        If true, specifies ouput is is radians.
    dms: bool (optional)
        If true, output is in degrees/minutes/seconds. Incompatible
        with rads option.
    Returns
    -------
    numpy.ndarray
        GPS (i.e. WGS84 datum) latitude value(s).
    numpy.ndarray
        GPS (i.e. WGS84 datum) longitude value(s).
    Examples
    --------
    >>> get_gps_lat_long_from_easting_northing([429157], [623009])
    (array([55.5]), array([-1.540008]))
    References
    ----------
    Based on the formulas in "A guide to coordinate systems in Great Britain".
    See also https://webapps.bgs.ac.uk/data/webservices/convertForm.cfm
    """
    
    phi = []
    lam = []
    east = array(east)
    north = array(north)
    
    for i in range(len(east)):
        phi_p = ((north[i] - osgb36.N_0)/(wgs84.a * wgs84.F_0)) + osgb36.phi_0
    
        phi_0 = osgb36.phi_0
        n = wgs84.n
        M = get_M(phi_p, phi_0, n)
        diff = north[i] - osgb36.N_0 - M
        
        while absolute(diff) >= 0.01:
            phi_p = ((north[i] - osgb36.N_0 - M)/ (wgs84.a * wgs84.F_0)) + phi_p
            M = get_M(phi_p, phi_0, n)
            diff = north[i] - osgb36.N_0 - M

        v = wgs84.a * wgs84.F_0 * (1 - wgs84.e2 * sin(phi_p)**2) ** -0.5

        rho = wgs84.a * wgs84.F_0 * (1 - wgs84.e2) * (1 - wgs84.e2 * sin(phi_p)**2) ** -1.5

        eta2 = (v / rho) - 1
    
        VII = tan(phi_p) / (2 * rho * v)

        VIII = (tan(phi_p) / (24 * rho * v**3)) * ( 5 + (3 * tan(phi_p)**2) + eta2 - (9 * tan(phi_p)**2 * eta2))

        IX = (tan(phi_p) / (720 * rho * v**5)) * ( 61 + (90 * tan(phi_p)**2) + (45 * tan(phi_p)**4))

        X = (1 / cos(phi_p)) / v 

        XI = (1 / cos(phi_p)) / (6 * v**3) * ((v/rho) + 2 * tan(phi_p)**2) 

        XII = (1 / cos(phi_p)) / (120 * v**5) * ( 5 + 28 * tan(phi_p)**2 + 24 * tan(phi_p)**4) 
        
        XIIA = (1 / cos(phi_p)) / (5040 * v**7) * ( 61 + 662 * tan(phi_p)**2 + 1320 * tan(phi_p)**4 + 720 * tan(phi_p)**6) 
    
        p = phi_p - VII * (east[i] - osgb36.E_0)**2 + VIII * (east[i] - osgb36.E_0)**4 - IX * (east[i] - osgb36.E_0)**6
        l = osgb36.lam_0 + X * (east[i] - osgb36.E_0) - XI * (east[i] - osgb36.E_0)**3 + XII * (east[i] - osgb36.E_0)**5 - XIIA * (east[i] - osgb36.E_0)**7
        
        phi.append(p)
        lam.append(l)

    phi = array(phi)
    lam = array(lam)
    
    if rads and dms:
        raise Exception('Both rads and dms cannot be true')
    if rads and dms==False:
        return phi, lam
    elif dms and rads==False:
        return rad2dms(phi), rad2dms(lam)
    else:
        return rad2deg(phi), rad2deg(lam) 


class HelmertTransform(object):
    """Callable class to perform a Helmert transform."""
    
    def __init__(self, s, rx, ry, rz, T):

        self.T = T.reshape((3, 1))
        
        self.M = array([[1+s, -rz, ry],
                        [rz, 1+s, -rx],
                        [-ry, rx, 1+s]])

    def __call__(self, X):
        X = X.reshape((3,-1))
        return self.T + self.M@X

class HelmertInverseTransform(object):
    """Callable class to perform the inverse of a Helmert transform."""
    
    def __init__(self, s, rx, ry, rz, T):

        self.T = T.reshape((3, 1))
        
        self.M = inv(array([[1+s, -rz, ry],
                        [rz, 1+s, -rx],
                        [-ry, rx, 1+s]]))

    def __call__(self, X):
        X = X.reshape((3,-1))
        return self.M@(X-self.T)

OSGB36transform = HelmertTransform(20.4894e-6,
                             -dms2rad(0,0,0.1502),
                             -dms2rad(0,0,0.2470),
                             -dms2rad(0,0,0.8421),
                             array([-446.448, 125.157, -542.060]))

WGS84transform = HelmertInverseTransform(20.4894e-6,
                             -dms2rad(0,0,0.1502),
                             -dms2rad(0,0,0.2470),
                             -dms2rad(0,0,0.8421),
                             array([-446.448, 125.157, -542.060]))


def WGS84toOSGB36(phi, lam, rads=False):
    """Convert WGS84 latitude/longitude to OSGB36 latitude/longitude.
    
    Parameters
    ----------
    phi : array_like or float
        Latitude in degrees or radians on WGS84 datum.
    lam : array_like or float
        Longitude in degrees or radians on WGS84 datum.
    rads : bool, optional
        If True, phi and lam are in radians. If False, phi and lam are in degrees.

    Returns
    -------
    tuple of numpy.ndarrays
        Latitude and longitude on OSGB36 datum in degrees or radians.
    """
    xyz = OSGB36transform(lat_long_to_xyz(asarray(phi), asarray(lam),
                                  rads=rads, datum=wgs84))
    return xyz_to_lat_long(*xyz, rads=rads, datum=osgb36)

def OSGB36toWGS84(phi, lam, rads=False):
    """Convert OSGB36 latitude/longitude to WGS84 latitude/longitude.
    
    Parameters
    ----------
    phi : array_like or float
        Latitude in degrees or radians on OSGB36 datum.
    lam : array_like or float
        Longitude in degrees or radians on OSGB36 datum.
    rads : bool, optional
        If True, phi and lam are in radians. If False, phi and lam are in degrees.

    Returns
    -------
    tuple of numpy.ndarrays
        Latitude and longitude on WGS84 datum in degrees or radians.
    """
    xyz = WGS84transform(lat_long_to_xyz(asarray(phi), asarray(lam),
                                  rads=rads, datum=osgb36))
    return xyz_to_lat_long(*xyz, rads=rads, datum=wgs84)

