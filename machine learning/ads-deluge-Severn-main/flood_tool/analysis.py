"""Analysis tools."""

import os
import math

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

__all__ = ['plot_risk_map']

def plot_postcode_density(postcode_file=(os.path.dirname(__file__)
                                      +'/resources/postcodes_unlabelled.csv'),
                          coordinate=['easting','northing'], dx=1000):
    """Plot a postcode density map from a postcode file."""

    
    pdb = pd.read_csv(postcode_file)

    bbox = (pdb[coordinate[0]].min()-0.5*dx, pdb[coordinate[0]].max()+0.5*dx,
            pdb[coordinate[1]].min()-0.5*dx, pdb[coordinate[1]].max()+0.5*dx)

    
    nx = (math.ceil((bbox[1] - bbox[0])/dx),
          math.ceil((bbox[3] - bbox[2])/dx))

    x = np.linspace(bbox[0]+0.5 * dx, bbox[0] + (nx[0] - 0.5) * dx, nx[0])
    y = np.linspace(bbox[2]+0.5 * dx, bbox[2] + (nx[1] - 0.5) * dx, nx[1])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(nx, int)

    for x, y in pdb[coordinate].values:
        Z[math.floor((x-bbox[0])/dx), math.floor((y-bbox[2])/dx)] += 1

    plt.pcolormesh(X, Y, np.where(Z>0, Z, np.nan).T,
                   norm=matplotlib.colors.LogNorm())
    plt.axis('equal')
    plt.colorbar()

def plot_risk_map(risk_data, coordinate=['easting','northing'], dx=1000):
    """Plot a risk map."""

    bbox = (risk_data[coordinate[0]].min()-0.5*dx, risk_data[coordinate[0]].max()+0.5*dx,
            risk_data[coordinate[1]].min()-0.5*dx, risk_data[coordinate[1]].max()+0.5*dx)

    
    nx = (math.ceil((bbox[1] - bbox[0])/dx),
          math.ceil((bbox[3] - bbox[2])/dx))

    x = np.linspace(bbox[0]+0.5 * dx, bbox[0] + (nx[0] - 0.5) * dx, nx[0])
    y = np.linspace(bbox[2]+0.5 * dx, bbox[2] + (nx[1] - 0.5) * dx, nx[1])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(nx, int)

    for x, y, val in risk_data[coordinate+'risk'].values:
        Z[math.floor((x-bbox[0])/dx), math.floor((y-bbox[2])/dx)] += val

    plt.pcolormesh(X, Y, np.where(Z>0, Z, np.nan).T,
                   norm=matplotlib.colors.LogNorm())
    plt.axis('equal')
    plt.colorbar()
        
        
    
