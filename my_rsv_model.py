from particles.state_space_models import StateSpaceModel
from particles.distributions import Normal
from particles import distributions as dists
import numpy as np


from particles.state_space_models import StateSpaceModel
from particles.distributions import Normal
import numpy as np

class RSVModel(StateSpaceModel):

    def __init__(self, mu=0.0, phi=0.95, sigma_eta2=0.1, xi=0.0, sigma_u2=0.1):
        """Initialise le modèle avec des paramètres spécifiques."""
        self.mu = mu
        self.phi = phi
        self.sigma_eta2 = sigma_eta2
        self.xi = xi
        self.sigma_u2 = sigma_u2

    def PX0(self):
        """Distribution initiale pour x_1."""
        variance = self.sigma_eta**2 / (1 - self.phi**2)
        return Normal(loc=self.mu, scale=np.sqrt(variance))
    
    def PX(self, t, xp):
        """Transition d'état x_t+1 | x_t."""
        mean = self.mu + self.phi * (xp - self.mu)
        return Normal(loc=mean, scale=self.sigma_eta)
    
    def PY(self, t, xp, x):
        """Observation y_t | x_t."""
        mean_y1 = 0  # Hypothèse de rendement moyen nul
        std_y1 = np.exp(x / 2)  # Déviation standard de y1
        log_vol = x + self.xi  # Log-volatilité (y2)
        return Normal(loc=[mean_y1, log_vol], scale=[std_y1, self.sigma_u])
