import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from dataclasses import dataclass

@dataclass
class RingParameters:
    """Parameters for the growing ring problem"""
    mu: float  # shear modulus
    A0: float  # inner radius in reference configuration
    B0: float  # outer radius in reference configuration
    gamma_r: float  # radial growth factor
    gamma_theta: float  # circumferential growth factor
    P: float  # internal pressure

class GrowingRingSolver:
    def __init__(self, params: RingParameters):
        self.p = params
        
    def growth_function(self, R0):
        """Compute g(R0) = γr * γθ"""
        return self.p.gamma_r * self.p.gamma_theta
    
    def compute_r(self, R0, a):
        """Compute current radius r using equation 13.50"""
        def integrand(rho):
            return self.growth_function(rho) * rho
        
        result = quad(integrand, self.p.A0, R0)
        return np.sqrt(a**2 + 2*result[0])
    
    def compute_stretches(self, R0, r):
        """Compute elastic stretches αr and αθ using equation 13.56"""
        alpha_r = (R0 * self.p.gamma_theta) / r
        alpha_theta = r / (R0 * self.p.gamma_theta)
        return alpha_r, alpha_theta
    
    def compute_stress_integrand(self, R0, a):
        """Compute integrand for τ in equation 13.53"""
        r = self.compute_r(R0, a)
        alpha_r, alpha_theta = self.compute_stretches(R0, r)
        
        # Neo-Hookean derivatives (equation 13.59)
        dW_dalpha_r = self.p.mu * alpha_r
        dW_dalpha_theta = self.p.mu * alpha_theta
        
        # Compute stress difference (t_θ - t_r)
        stress_diff = alpha_theta * dW_dalpha_theta - alpha_r * dW_dalpha_r
        
        return stress_diff / (r**2) * self.growth_function(R0) * R0
    
    def compute_tau(self, A0, R0, a):
        """Compute τ using equation 13.53"""
        result = quad(lambda x: self.compute_stress_integrand(x, a), A0, R0)
        return result[0]
    
    def find_bracket_for_a(self):
        """Find the correct value for 'a' that corresponds to the boundary condition 'P'"""
        a_min = -10.0 * self.p.A0 * np.sqrt(self.p.gamma_r * self.p.gamma_theta)
        a_max = 1000.0 * self.p.A0 * np.sqrt(self.p.gamma_r * self.p.gamma_theta)
        
        def objective(a):
            return self.compute_tau(self.p.A0, self.p.B0, a) - self.p.P
        
        # Test various points to find bracket
        test_points = np.linspace(a_min, a_max, 10000)
        values = [objective(a) for a in test_points]
        
        # When the sign changes, you have a max and min value for a that corresponds to P
        for i in range(len(values)-1):
            if values[i] * values[i+1] <= 0:
                return test_points[i], test_points[i+1]
                
        raise ValueError(f"Could not find an 'a' that corresponds to P. Function values: {values}")
    
    def find_inner_radius(self):
        """Find inner radius 'a' by solving equation 13.54"""
        def objective(a):
            return self.compute_tau(self.p.A0, self.p.B0, a) - self.p.P
        
        # Find appropriate bracket
        a_left, a_right = self.find_bracket_for_a()
        
        print(f"Found bracket: [{a_left:.6f}, {a_right:.6f}]")
        print(f"Function values at bracket: [{objective(a_left):.6f}, {objective(a_right):.6f}]")
        
        # Use root finding to find the correct value for 'a'
        result = root_scalar(objective, bracket=[a_left, a_right])
        print(f"Found solution a = {result.root:.6f}")
        return result.root
    
    def compute_stresses(self, R0_points):
        """Compute radial and hoop stresses at given points"""
        a = self.find_inner_radius()
        
        t_r = np.zeros_like(R0_points)
        t_theta = np.zeros_like(R0_points)
        
        for i, R0 in enumerate(R0_points):
            # Compute radial stress using equation 13.57
            t_r[i] = -self.p.P + self.compute_tau(self.p.A0, R0, a)
            
            # Compute current radius and stretches
            r = self.compute_r(R0, a)
            alpha_r, alpha_theta = self.compute_stretches(R0, r)
            
            # Compute hoop stress using equation 13.58
            dW_dalpha_r = self.p.mu * alpha_r
            dW_dalpha_theta = self.p.mu * alpha_theta
            t_theta[i] = t_r[i] + alpha_theta * dW_dalpha_theta - alpha_r * dW_dalpha_r
            
        return t_r, t_theta

if __name__ == "__main__":
    # Parameters for Figure 13.5A
    params = RingParameters(
        mu=1.0,
        A0=1.0,
        B0=2.0,
        gamma_r=1.1,
        gamma_theta=1.0,
        P=0.0
    )
    
    solver = GrowingRingSolver(params)
    R0_points = np.linspace(params.A0, params.B0, 100)
    t_r, t_theta = solver.compute_stresses(R0_points)
    
    # Plot results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(R0_points, t_r, label='Radial stress')
    plt.plot(R0_points, t_theta, label='Hoop stress')
    plt.xlabel('R0')
    plt.ylabel('Stress')
    plt.legend()
    plt.grid(True)
    plt.title('Stresses in Growing Ring (Fig 13.5A)')
    plt.savefig('growing_ring_stresses.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Plot saved to 'growing_ring_stresses.png'")