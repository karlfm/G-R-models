import ufl

'''Incompressible part'''
def holzapfel(F):
    """

    Declares the strain energy function for a simplified holzapfel formulation.

    Args:
        F - deformation tensor

    Returns:
        psi(F), scalar function

    """

    a = 0.074
    b = 4.878
    a_f = 2.628
    b_f = 5.214

    J = ufl.det(F)
    C = pow(J, -float(2) / 3) * F.T * F
 
    e1 = ufl.as_vector([1.0, 0.0, 0.0])

    IIFx = ufl.tr(C)
    I4e1 = ufl.inner(C * e1, e1)

    cond = lambda a: ufl.conditional(a > 0, a, 0)

    W_hat = a / (2 * b) * (ufl.exp(b * (IIFx - 3)) - 1)
    W_f = a_f / (2 * b_f) * (ufl.exp(b_f * cond(I4e1 - 1) ** 2) - 1)

    return  W_hat + W_f

def hookean(w, C):
    return w*C

def neohookean(w, C):
    """

    Declares the strain energy function for a neohookean (incompressible) material.

    Args:
        F - deformation tensor

    Returns:
        psi(F), scalar function

    """

    I1 = ufl.tr(C)

    return w*(I1 - 3)

def mooney_rivlin(w1, w2, C):
    """
    Args:
        w1, w2 - weights
        C - Ratio tensor (Cauchy tensor)

    Returns:
        psi(C), scalar function

    """
    
    I1 = ufl.tr(C)
    I2 = ufl.tr(ufl.inv(C))*ufl.det(C)
    return w1/2*(I1 - 3) + w2/2*(I2 - 3)

def fung_demiray(mu, beta, C):
    """
    Args:
        mu - shear modulus
        beta - controls the strain hardening property
        C - Ratio (Cauchy) tensor

    Returns:
        psi(C), scalar function

    """
    
    I1 = ufl.tr(C)
    return mu/(2*beta)*(ufl.exp(beta*(I1 - 3)) - 1)

def gent(mu, beta, C):
    """
    Args:
        mu - shear modulus
        beta - controls the strain hardening property
        C - Ratio (Cauchy) tensor

    Returns:
        psi(C), scalar function

    """
    
    I1 = ufl.tr(C)
    return -mu/(2*beta)*(ufl.ln(1 - beta*(I1 - 3)))

def guccione(w1, w2, w3, w4, C):
    """
    Args:
        mu - shear modulus
        beta - controls the strain hardening property
        C - Ratio (Cauchy) tensor

    Returns:
        psi(C), scalar function

    """

    E = 1/2*(C - ufl.Identity(3))
    Q = w1*E[0,0]**2 + w2*(E[1,1]**2 + E[2,2]**2 + E[1,2]*E[2,1]) + w3*(2*E[0,1]*E[1,0] + 2*E[0,2]*E[2,0])
    
    return 1/2*w4*(ufl.exp(Q) - 1)

'''Compressible part'''
def comp1(mu_c , J):
    I3 = pow(J, 2)
    return mu_c*(I3 - 1)

def comp2(mu_c , J):
    return mu_c*pow((J - 1), 2)

def comp3(mu_c , J):
    I3 = pow(J, 2)
    return mu_c*ufl.ln(I3)

def comp4(mu_c , J):
    return mu_c*ufl.ln(J)

def comp5(mu_1, mu_2, J):
    I3 = pow(J, 2)
    return -mu_1/2*(I3 - 1) + mu_2/4*(pow((I3 - 1), 2))

def comp6(w1, J):
    I3 = pow(J, 2)
    return w1 * (J * ufl.ln(J) - J + 1)*ufl.Identity(3)

def comp7(w, J):
    return 0.5*w*(J-1)*ufl.ln(J)

def main():
    return

if __name__ == "__main__":
    main()