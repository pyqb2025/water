# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How much water is there?

# %%
import math
from urllib.request import Request, urlopen
import numpy as np
from io import BytesIO
from matplotlib import image
import matplotlib.pyplot as plt


# %%
def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Converts geographic coordinates (latitude and longitude) into x, y tile coordinates for OSM.
    
    Note:
        - Latitude (lat) must be between -85.0511 and 85.0511 degrees.
          This limit is due to the Mercator projection used by OpenStreetMap
          https://en.wikipedia.org/wiki/Web_Mercator_projection,
          which cannot represent the poles (±90 degrees).
        - Longitude (lon) must be between -180 and 180 degrees.
    
    Returns:
        tuple[int, int]: Tile coordinates
        
    """
    x = int((lon + 180.0) / 360.0 * (2**zoom))
    y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2.0 * (2**zoom))
    return x, y


def is_sea_tile(x: int, y: int, zoom: int = 14) -> tuple[bool, np.ndarray]:
    """Get a OSM tile and check if is covered by sea.

    Examples:
        >>> x, y = lat_lon_to_tile(45+28/60+1/60**2, 9+11/60+24/60**2, 14) # Milano
        >>> b, _ = is_sea_tile(x, y, 14)
        >>> b is False # Not covered by sea
        True
        >>> x, y = lat_lon_to_tile(11+21/60, 142+12/60, 14) # Fossa delle Marianne
        >>> b, _ = is_sea_tile(x, y, 14)
        >>> b is True # Covered by sea
        True

    """
    # Tile URL
    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    
    # https://operations.osmfoundation.org/policies/tiles/
    headers = {
        "User-Agent": "ProgrammingInPythonCourse/1.0 (contact: mattiamonga+osm@gmail.com)"
    }

    with urlopen(Request(url, headers=headers)) as tile:
        if tile.status != 200:
            raise ValueError(f"Impossibile scaricare il tile: {url}")
        data = tile.read()
    
    # Load the image
    img = image.imread(BytesIO(data))
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    return bool(((b > r) & (b > g)).mean() > 1/2), img


# %%
rng = np.random.default_rng(seed=16122025)  # Random number generator

imgs = []
data = []

minlat = -85.0511  # Minimum latitude
maxlat = -minlat  # Maximum latitude
minlong = -180.0  # Minimum longitude
maxlong = -minlong  # Maximum longitude
zoom = 14  # Zoom level

for i in range(30):
    lat = (maxlat - minlat) * rng.random() + minlat  # Random latitude
    long = (maxlong - minlong) * rng.random() + minlong  # Random longitude
    
    x, y = lat_lon_to_tile(lat, long, zoom)
    b, img = is_sea_tile(x, y, zoom)
    imgs.append(img)
    if b:
        data.append('W')
    else:
        data.append('L')


# %%
fig, ax = plt.subplots(ncols=len(imgs))  # Create subplots for visualization
for i, img in enumerate(imgs):
    ax[i].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax[i].imshow(img)  # Display the image
    ax[i].set_title(data[i])  # Set the title as 'W' or 'L'


# %%
def dbinom(successes: int, n: int, prob: float | np.ndarray) -> float | np.ndarray:
    """ The probability of a number of successes for an event with probability prob repeated n times. 
    See: https://en.wikipedia.org/wiki/Binomial_distribution
    
    >>> math.isclose(dbinom(4, 4, 0.25), 0.25**4)
    True

    >>> bool(np.isclose(dbinom(4, 4, np.array([0, 0.25, 1])), np.array([0, .25**4, 1])).all())
    True
    """

    failures = n - successes
    return math.factorial(n)/(math.factorial(successes)*math.factorial(failures))*prob**successes*(1-prob)**(failures)



# %%
W = sum([1 if x == 'W' else 0 for x in data])
L = len(data) - W


granularity = 50  # Number of points in the grid (proportion hypotheses considered)
p_grid = np.linspace(start=0, stop=1, num=granularity)

prior_lst = [] # Define the prior probability distribution
for p in p_grid:
    if p < 0.2:
        prior_lst.append(0)
    else:
        prior_lst.append(1)
prior = np.array(prior_lst) / sum(prior_lst) # normalize: must sum to one

likelihood = dbinom(W, n=W+L, prob=p_grid)
unstd_posterior = likelihood * prior
posterior = unstd_posterior / unstd_posterior.sum() # normalize: must sum to one

# %%
fig, ax = plt.subplots()  # Create a plot for prior, likelihood, and posterior

ax.plot(p_grid, prior, label='prior')
ax.plot(p_grid, likelihood, label='likelihood')
ax.plot(p_grid, posterior, label='posterior')
ax.set_xlabel('Proportion of Water')
ax.set_ylabel('Probability')

_ = ax.legend()

# %%
import pymc as pm # type: ignore

# %%
with pm.Model() as water_model:
    p = pm.Uniform('p', 0.2, 1)
    w = pm.Binomial('W', observed=W, n=W+L, p=p)

# %%
water_model

# %%
with water_model:
    idata = pm.sample(seed=20251216)

# %%
import arviz as az

# %%
az.summary(idata)

# %%
_ = az.plot_posterior(idata, hdi_prob=.99)
