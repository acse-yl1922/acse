##########
Flood Tool
##########

This package implements a flood risk prediction and visualization tool.

Installation Instructions
-------------------------

To install, run the command below in a terminal:
``python setup.py install``

The package contained in the requirements.txt file should be pip installed by running the following command:
``pip install -r requirements.txt``

The environment can be installed through the following command:
``conda env create -f environment.yml``


Usage guide
-----------

Modelling Instructions
----------------------

The task of modelling involved predicting respective variables ranging from flood probability, median price and local authority. Flood probability had features pertaining to latitude, longitude, soil type and altitude; median price used latitude, longitude, soil type, altitude and local authority; and local authority capitalised on only easting and northing. Several options were considered for the model, however the best ones were Random Forest Regressor for flood probability, and KNeighborsRegressor for median price and local authority. It is worth noting that an additional model was made to predict the flood probability but alternatively only using the latitude and longitude which makes use of Random Forest Regressor. This fourth model is an alternative to the first model and it also gives a similar score.

There were several python files created for the workflow of the four models which are: 
floodclass_from_loc_model.py
floodclassmodel.py 
localauthoritymodel.py
median_price_model.py

However, kindly note that the tool.py pertains to the final training of the dataset and is somewhat a compilation of the four python files mentioned above. There is a notebook / interface called ```toolUserInterface.ipynb``` which contains examples of how to run the tool file to generate predictions 

Visualisation Instructions
--------------------------

For the visualisation, the main module is "visualisation.py". Users can import this module with the class Visualisation. When the Visualisation is initialised (eg. vis=Visualisation), it automatically reads the some preexisted rainfall, river stage and tidal data files from the folder "flood_tool/resources". Users can also change the file path to what they want to show different data.

If users want to plot the risk of flood trained by models, they need to call the function "plotfile(datapath)" in this class, the data can be any data with the same structure as the training datasets("flood_tool/resources/postcodes_sampled_data.csv").

After calling the plotfile function (eg. vis.plotfile()), call the map attribute of the object (eg. vis.map)the map will show with the default layer risk of flood on. Map can be dragged and zoomed. In order to see more layers like rainfall or river data. Just click the layer icon on the right corner, and select or deselect the layer to display on or off.

For the individual point information on the map, just move mouse over any maker or icon, more information like values or position will show up.

For the display of the live data, first, user need to use function "get_all_live_data" in "live.py" file, which will update the data file "flood_tool/resources/latest_live_data.csv". Updating process will take about 15 - 20 minutes, so user can just plot the data in this file directly without updating the data for testing. "plotfile() function will automatically plot the live data based on the "latest_live_data.csv".


Geodetic Transformations
------------------------

For historical reasons, multiple coordinate systems exist in British mapping.
The Ordnance Survey has been mapping the British Isles since the 18th Century
and the last major retriangulation from 1936-1962 produced the Ordance Survey
National Grid (or **OSGB36**), which defined latitude and longitude across the
island of Great Britain [1]_. For convenience, a standard Transverse Mercator
projection [2]_ was also defined, producing a notionally flat gridded surface,
with gradations called eastings and westings. The scale for these gradations
was identified with metres.


The OSGB36 datum is based on the Airy Ellipsoid of 1830, which defines
semimajor axes for its model of the earth, :math:`a` and :math:`b`, a scaling
factor :math:`F_0` and ellipsoid height, :math:`H`.

.. math::
    a &= 6377563.396, \\
    b &= 6356256.910, \\
    F_0 &= 0.9996012717, \\
    H &= 24.7.

The point of origin for the transverse Mercator projection is defined in the
Ordnance Survey longitude-latitude and easting-northing coordinates as

.. math::
    \phi^{OS}_0 &= 49^\circ \mbox{ north}, \\
    \lambda^{OS}_0 &= 2^\circ \mbox{ west}, \\
    E^{OS}_0 &= 400000 m, \\
    N^{OS}_0 &= -100000 m.

More recently, the world has gravitated towards the use of Satellite based GPS
equipment, which uses the (globally more appropriate) World Geodetic System
1984 (or **WGS84**). This datum uses a different ellipsoid, which offers a
better fit for a global coordinate system. Its key properties are:

.. math::
    a_{WGS} &= 6378137,, \\
    b_{WGS} &= 6356752.314, \\
    F_0 &= 0.9996.

For a given point on the WGS84 ellipsoid, an approximate mapping to the
OSGB36 datum can be found using a Helmert transformation [3]_,

.. math::
    \mathbf{x}^{OS} = \mathbf{t}+\mathbf{M}\mathbf{x}^{WGS}.


Here :math:`\mathbf{x}` denotes a coordinate in Cartesian space (i.e in 3D)
as given by the (invertible) transformation

.. math::
    \nu &= \frac{aF_0}{\sqrt{1-e^2\sin^2(\phi^{OS})}} \\
    x &= (\nu+H) \sin(\lambda)\cos(\phi) \\
    y &= (\nu+H) \cos(\lambda)\cos(\phi) \\
    z &= ((1-e^2)\nu+H)\sin(\phi)

and the transformation parameters are

.. math::
    :nowrap:

    \begin{eqnarray*}
    \mathbf{t} &= \left(\begin{array}{c}
    -446.448\\ 125.157\\ -542.060
    \end{array}\right),\\
    \mathbf{M} &= \left[\begin{array}{ c c c }
    1+s& -r_3& r_2\\
    r_3 & 1+s & -r_1 \\
    -r_2 & r_1 & 1+s
    \end{array}\right], \\
    s &= 20.4894\times 10^{-6}, \\
    \mathbf{r} &= [0.1502'', 0.2470'', 0.8421''].
    \end{eqnarray*}

Given a latitude, :math:`\phi^{OS}` and longitude, :math:`\lambda^{OS}` in the
OSGB36 datum, easting and northing coordinates, :math:`E^{OS}` & :math:`N^{OS}`
can then be calculated using the following formulae:

.. math::
    \rho &= \frac{aF_0(1-e^2)}{\left(1-e^2\sin^2(\phi^{OS})\right)^{\frac{3}{2}}} \\
    \eta &= \sqrt{\frac{\nu}{\rho}-1} \\
    M &= bF_0\left[\left(1+n+\frac{5}{4}n^2+\frac{5}{4}n^3\right)(\phi^{OS}-\phi^{OS}_0)\right. \\
    &\quad-\left(3n+3n^2+\frac{21}{8}n^3\right)\sin(\phi-\phi_0)\cos(\phi^{OS}+\phi^{OS}_0) \\
    &\quad+\left(\frac{15}{8}n^2+\frac{15}{8}n^3\right)\sin(2(\phi^{OS}-\phi^{OS}_0))\cos(2(\phi^{OS}+\phi^{OS}_0)) \\
    &\left.\quad-\frac{35}{24}n^3\sin(3(\phi-\phi_0))\cos(3(\phi^{OS}+\phi^{OS}_0))\right] \\
    I &= M + N^{OS}_0 \\
    II &= \frac{\nu}{2}\sin(\phi^{OS})\cos(\phi^{OS}) \\
    III &= \frac{\nu}{24}\sin(\phi^{OS})cos^3(\phi^{OS})(5-\tan^2(phi^{OS})+9\eta^2) \\
    IIIA &= \frac{\nu}{720}\sin(\phi^{OS})cos^5(\phi^{OS})(61-58\tan^2(\phi^{OS})+\tan^4(\phi^{OS})) \\
    IV &= \nu\cos(\phi^{OS}) \\
    V &= \frac{\nu}{6}\cos^3(\phi^{OS})\left(\frac{\nu}{\rho}-\tan^2(\phi^{OS})\right) \\
    VI &= \frac{\nu}{120}\cos^5(\phi^{OS})(5-18\tan^2(\phi^{OS})+\tan^4(\phi^{OS}) \\
    &\quad+14\eta^2-58\tan^2(\phi^{OS})\eta^2) \\
    E^{OS} &= E^{OS}_0+IV(\lambda^{OS}-\lambda^{OS}_0)+V(\lambda-\lambda^{OS}_0)^3+VI(\lambda^{OS}-\lambda^{OS}_0)^5 \\
    N^{OS} &= I + II(\lambda^{OS}-\lambda^{OS}_0)^2+III(\lambda-\lambda^{OS}_0)^4+IIIA(\lambda^{OS}-\lambda^{OS}_0)^6



Function APIs
-------------

.. automodule:: flood_tool
  :members:
  :imported-members:


.. rubric:: References

.. [1] A guide to coordinate systems in Great Britain, Ordnance Survey
.. [2] Map projections - A Working Manual, John P. Snyder, https://doi.org/10.3133/pp1395
.. [3] Computing Helmert transformations, G Watson, http://www.maths.dundee.ac.uk/gawatson/helmertrev.pdf
