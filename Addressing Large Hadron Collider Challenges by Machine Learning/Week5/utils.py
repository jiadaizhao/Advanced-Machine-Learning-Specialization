import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



class Tracker(object):
    
    def __init__(self, R, pitch, y1, y2, y3, z1, z2, z3):
        """
        Generates Z, Y coordinates of straw tubes of the tracking system.

        Parameters:
        -----------
        R : float
            Radius of a straw tube.
        pitch : float
            Distance between two adjacent tubes in one layer of the system.
        y1 : float
            Shift between two layers of tubes.
        y2 : float
            Shift between two layers of tubes.
        y3 : float
            Shift between two layers of tubes.
        z1 : float
            Shift between two layers of tubes.
        z2 : float
            Shift between two layers of tubes.
        z3 : float
            Shift between two layers of tubes.
        """
        
        self.R = R
        self.pitch = pitch
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.z1 = z1
        self.z2 = z2
        self.z3 = z3
        
        
    def create_geometry(self):
        """
        Generate Z, Y coordinates of the tubes.
        """
        
        base = np.arange(-100, 101, 1)
        step = self.pitch
        
        layer1_y = step * base
        layer1_z = 0. * np.ones(len(base))
        
        layer2_y = layer1_y + self.y1
        layer2_z = layer1_z + self.z1
        
        layer3_y = layer2_y + self.y2
        layer3_z = layer2_z + self.z2
        
        layer4_y = layer3_y + self.y3
        layer4_z = layer3_z + self.z3
        
        Z = np.concatenate((layer1_z.reshape(-1, 1), 
                            layer2_z.reshape(-1, 1), 
                            layer3_z.reshape(-1, 1), 
                            layer4_z.reshape(-1, 1)), axis=1)
        
        Y = np.concatenate((layer1_y.reshape(-1, 1), 
                            layer2_y.reshape(-1, 1), 
                            layer3_y.reshape(-1, 1), 
                            layer4_y.reshape(-1, 1)), axis=1)
        
        geo = [Z, Y]
        
        return geo

    
def geometry_display(Z, Y, R, y_min=-10, y_max=10):
    """
    Displays straw tubes of the tracking system.

    Parameters:
    -----------
    Z : array_like
        Array of z-coordinates of the tubes.
    Y : array_like
        Array of y-coordinates of the tubes.
    R : float
        Radius of a tube.
    y_min : float
        Minimum y-coordinate to display.
    y_max : float
        Maximum y-coordinate to display.
    """

    Z_flat = np.ravel(Z)
    Y_flat = np.ravel(Y)

    z_min = Z_flat.min()
    z_max = Z_flat.max()

    sel = (Y_flat >= y_min) * (Y_flat < y_max)
    Z_flat = Z_flat[sel]
    Y_flat = Y_flat[sel]

    plt.figure(figsize=(8, 8 * (y_max - y_min + 2) / (z_max - z_min + 10)))
    plt.scatter(Z_flat, Y_flat)

    for z,y in zip(Z_flat, Y_flat):
        circle = plt.Circle((z, y), R, color='b', fill=False)
        plt.gcf().gca().add_artist(circle)

    plt.xlim(z_min - 5, z_max + 5)
    plt.ylim(y_min - 1, y_max + 1)
    plt.xlabel('Z', size=14)
    plt.ylabel('Y', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    
    

    
class Tracks(object):
    
    def __init__(self, b_min, b_max, alpha_mean, alpha_std):
        """
        Generates tracks.

        Parameters:
        -----------
        b_min : float
            Minimum y intercept of tracks.
        b_max : float
            Maximum y intercept of tracks.
        alpha_mean : float
            Mean value of track slopes.
        alpha_std : float
            Standard deviation of track slopes.
        """
        
        self.b_min = b_min
        self.b_max = b_max
        self.alpha_mean = alpha_mean
        self.alpha_std = alpha_std
        
    def generate(self, N):
        """
        Generates tracks.

        Parameters:
        -----------
        N : int
            Number of tracks to generate.

        Returns:
        --------
        tracks : array-like
            List of track parameters [[k1, b1], [k2, b2], ...]
        """
        
        B = np.random.RandomState(42).uniform(self.b_min, self.b_max, N)
        Angles = np.random.RandomState(42).normal(self.alpha_mean, self.alpha_std, N)
        K = np.tan(Angles)
        
        tracks = np.concatenate((K.reshape(-1, 1), B.reshape(-1, 1)), axis=1)
        
        return tracks
    
    
def tracks_display(tracks, Z):
    """
    Displays tracks.

    Parameters:
    -----------
    tracks : array-like
        List of track parameters.
    Z : array-like
        List of z-coordinates.
    """

    Z_flat = np.ravel(Z)

    z_min = Z_flat.min()
    z_max = Z_flat.max()
    
    z1 = z_min - 5
    z2 = z_max + 5

    for k, b in tracks:
        
        plt.plot([z1, z2], [k * z1 + b, k * z2 + b], c='0.2', alpha=0.3)


        
def get_score(Z, Y, tracks, R):
    """
    Score of the tracking system geometry.
    Z : array_like
        Array of z-coordinates of the tubes.
    Y : array_like
        Array of y-coordinates of the tubes.
    R : float
        Radius of a tube.
    tracks : array-like
        List of track parameters.
    """
    
    values = []
    
    for k, b in tracks:
        
        Y_pred = k * Z + b
        dY = np.abs(Y_pred - Y)
        
        alpha = np.arctan(k)
        cos = np.cos(alpha)
        
        is_intersect = dY * cos < R
        n_intersections = (is_intersect).sum()
        
        if n_intersections >= 2:
            values.append(1)
        else:
            values.append(0)

    return np.mean(values)


def plot_objective(min_objective_values):
    """
    Plot optimization curve
    """
    plt.figure(figsize=(9, 6))
    plt.plot(min_objective_values, linewidth=2)
    plt.xlabel("Number of calls", size=14)
    plt.ylabel('Objective', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title('Optimization curve', loc='right', size=14)
    plt.grid(b=1)
    

from IPython.display import FileLink
def create_solution(best, filename='submission_file.csv'):
    """saves predictions to file and provides a link for downloading """
    df = pd.DataFrame(data=[best], columns=['R', 'pitch', 'y1', 'y2', 'y3', 'z1', 'z2', 'z3'])
    df.to_csv(filename, index_label=False, index=False)
    return FileLink('{}'.format(filename))