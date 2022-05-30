# -*- coding: utf-8 -*-
"""
PAnM-continuous-S: Model for continuous autotrophic growth of mixed
culture photoautotrophic and photoheterotrophic bacteria with sulfide
as electron donor. Part of publication in preparation:

F. Egger, T. Hulsen, D. J. Batstone,
Continuous H2S removal from biogas using purple phototrophic bacteria,
in preparation to be submitted to Water Research.

Model contains:
    - class parameter:
        dataclass defines all parameters for the model.
        Call paramters by using the dot notation e.g.:
            parameter.kla
    - function PAnMpH:
        calculating the pH for the main model
    - function PAnMchargeBalance:
        charge balance for the pH calcualtion
    - function PAnMsulfide:
        contains DAE system, balance equations, contitutive equations
    - function PAnMdynamicInput:
        defines dynamic input into the DAE system
    - function PAnMinitialConditions:
        defines initial conditions for the DAE system
    - function main:
        reads initial conditions and runs the DAE calculation.
Requires the 'numpy','scipy' and 'dataclasses' libraries.

Copyright (C) 2022  Felix Egger

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp


@dataclass
class parameter:
    """
    Parameter for the PAnM-continuous-S model and associated functions.
    """
    KaH2S: float = 10**-7
    KaCO2: float = 10**-6.35
    KaNH4: float = 10**-9.25
    KaH2PO4: float = 10**-7.21
    KaH2O: float = 10**-14
    KaHS: float = 10**-13
    KaH2SO4: float = 10**-2
    KaH2S2O3: float = 10**-1.74

    # mg/L
    KsSac: float = 20.22
    KsSS: float = 0.5242
    Ks0a: float = 1
    Ks0h: float = 50
    Ks2a: float = 10
    Ks2h: float = 50
    Ks3a: float = 200
    Ks3h: float = 100
    Ks1a: float = 100*32
    Ks1h: float = 100
    KsC: float = 5
    KIFA: float = 7850
    KsN: float = 0.02
    KsP: float = 0.081
    KsH2: float = 1
    pHLL: float = 5
    pHUL: float = 7
    # 1/h
    kHyd: float = 0
    kMac: float = 0
    kMph: float = 0
    kMch: float = 0
    k0a: float = 0.235  # 0.1
    k0h: float = 0.06
    kMh2: float = 0
    kDec: float = 0.09/24
    k3a: float = 0.4  # 0.23
    k3h: float = 0.001
    k1a: float = 19  # 4
    k1h: float = 0.35*32
    k2a: float = 0.16  # 0.032
    k2h: float = 0.001

    # 1/h
    klaIC: float = 4  # 0.55 #Experimental data
    klaIS: float = 4  # 0.55 #Experimental data
    klaN2: float = 0  # 0.003
    klaH2: float = 0  # 0.003
    klaCH4: float = 0.003  # 0.003

    # mol/(L*bar)
    KHic: float = 0.034
    KHis: float = 0.1
    KHn2: float = 0.0006
    KHh2: float = 0.00078
    KHch4: float = 0.0014

    rhoW: float = 1
    Vl: float = 2  # L
    Vg: float = 0.5  # L
    R: float = 0.083145  # bar L/(mol K)
    T: float = 297.15  # K
    P: float = 1.013  # bar
    kv: float = 100  # 2e6 #L/(h bar)

    fSsxs: float = 0.163824083
    fAcxs: float = 0.16683925
    fAcch: float = 0.6691
    fIcxs: float = 0.0156
    fIcch: float = 0.006402247
    fH2xs: float = 0.084424681
    fIsxs: float = 0
    fH2ch: float = 0.3309
    fInxs: float = 1.7907795E-03
    fSixpb: float = 1
    fIpxs: float = 0
    fXS0xs: float = 0
    fSso4xs: float = 0
    fSs2o3xs: float = 0
    fSixs: float = 0.151820854
    fXixs: float = 0.433091132

    Cm: float = 0.318427556
    Csi: float = 0.030148*12
    Nm: float = 14/180
    Nsi: float = 0.060026
    Pm: float = (14/180)/5
    Psi: float = 0.006489

    Ynet = 0.86
    YPBch: float = 0.7
    YPBph: float = 1
    YPB0a: float = 0.25*Ynet
    YPB0h: float = 0.25*Ynet
    YPBa: float = 0.4
    YPB1a: float = Ynet
    YPB1h: float = Ynet
    YPB2a: float = 0.5*Ynet
    YPB2h: float = 0.5*Ynet
    YPB3a: float = 0.25*Ynet
    YPB3h: float = 0.25*Ynet


def PAnMpH(yp, p):
    """
    Function to calculate pH in the PAnM-S model based on ADM1 species concentrations
    (Batstone et al. (2002)).
    Root finding algorithm uses a bracket from pH 1 to pH 14.
    Root finding method: toms748, using scipy.optimize library.
    Parameters
    ----------
    yp : tuple(float)
        Anions and cations to be provided to the cahrge balance
    p : dataclass
        Parameters used in the model.
    Returns
    -------
    pH : float
        pH corresponding to species concentrations
    """
    sol = root_scalar(
        PAnMchargeBalance, bracket=[10**-14, 10**-1], args=(yp, p), method='toms748')
    pH = - np.log10(sol.root)
    return pH


def PAnMchargeBalance(SH, yp, p):
    """
    Charge balance to calculate pH with all relevant cations and anions.
    Based on ADM1 (Batstone et al. (2002)).
    All concentrations in [mol/L].

    Parameters
    ----------
    SH : float
        Concentration of protons.
    yp : tuple(float)
        Concentrations of relevant ions: Cations (Cat), inorganic nitrogen (SIN),
        inorganic carbin (SIC), inorganic phosphorus (SIP), sulfate (SSO4),
        thiosulfate (SS2O3), anions (An)
    p : dataclass
        Parameters used in the model.
    Returns
    -------
    zerofun : float
        Root of the charge balance.

    """
    Cat, SIN, SIC, SIS, SIP, SSO4, SS2O3, An = yp

    zerofun = Cat \
        + SH*SIN/(p.KaNH4+SH) \
        + SH \
        - p.KaCO2*SIC/(p.KaCO2+SH) \
        - p.KaH2S*SIS/(p.KaH2S+SH) \
        - p.KaH2O/SH \
        - SH*SIP/(p.KaH2PO4+SH) \
        - 2*p.KaH2PO4*SIP/(p.KaH2PO4+SH) \
        - 2*SSO4 \
        - SS2O3 \
        - An
    return zerofun


def PAnMsulfide(t, y, p, u, tu):
    """
    Photo Anerobic model with addition for sulfur-containing substrates and
    products base on Egger et al. (2020).
    Autotrophic biomass and heterotrophic biomass with different biokinets
    considered.

    Parameters
    ----------
    t : float
        Current time.
    y : tuple(float)
        State variables.
    p : dataclass
        Parameters used in the model.

    Returns
    -------
    dydt : tuple(float)
        State variable derivatives.
    """

    # linear interpolation of input
    f = interp1d(tu, u, axis=0, fill_value="extrapolate")
    u = f(t)

    # Input vector assignement
    SSin, SAcin, SICin, SH2in, SISin, SINin, SIPin, XS0ain, XS0hin, SSO4in, \
        SS2O3in, SIin, XPBain, XPBhin, XSin, XIin, SN2in, pCO2in, pH2Sin, pN2in,\
        pH2in, Fliqin, Fgasin, T, pCH4in, SCH4in = u

    # Gas phase H2O saturation
    pH2O = 0.0313*np.exp(5290*(1/298-1/(T+273.15)))  # bar

    # State vector assignment
    SS, SAc, SIC, SH2, SIS, SIN, SIP, XS0a, XS0h, SSO4, SS2O3, SI, XPBa, XPBh, XS, XI, SN2, \
        pCO2, pH2S, pN2, pH2, SCH4, pCH4 = y

    ## pH
    # Cations and Anion concentrations
    Cat = 0.0135 #[mol/L]
    An = 0.0058 #[mol/L]
    # Ions participating in the charge balance in [mol/L]
    yp = np.array([Cat, SIN/14e3, SIC/12e3, SIS/64e3, SIP/31e3, SSO4/32e3,
                  SS2O3/32e3, An])
    pH = PAnMpH(yp, p)

    # Speciaction
    SHS = p.KaH2S*SIS/(p.KaH2S + 10**-pH)
    SH2Saq = SIS - SHS

    SHCO3 = p.KaCO2*SIC/(p.KaCO2 + 10**-pH)
    SCO2aq = SIC - SHCO3

    # Patching of the concentrations onto the function to bea ble to read it
    # in the main code if necessary
    PAnMsulfide.SHCO3 = SHCO3
    PAnMsulfide.SCO2aq = SCO2aq

    # Limitation models
    ISAc = SAc/(p.KsSac+SAc)
    ISS = SS/(p.KsSS+SS)
    IIC = SIC/(p.KsC+SIC)
    IIN = SIN/(p.KsN+SIN)
    IIP = 1  # SIP/(p.KsP+SIP) # Phosphate assumed not to be limiting
    IH2 = SH2/(p.KsH2 + SH2)
    IISa = SIS/(p.Ks0a+SIS)
    IISh = SIS/(p.Ks0h+SIS)
    I2a = SIS/(p.Ks2a+SIS)
    I2h = SIS/(p.Ks2h+SIS)
    ISS2O3a = SS2O3/(p.Ks3a+SS2O3)
    ISS2O3h = SS2O3/(p.Ks3h+SS2O3)
    IXS0a = XS0a/(p.Ks1a+XS0a)
    IXS0h = XS0h/(p.Ks1h+XS0h)

    # Inhibition by free nitric acid
    IFA = p.KIFA/(p.KIFA+SIN)

    # Competition models
    CISa = SIS/(SIS+XS0a+SS2O3+1e-4)
    CISh = SIS/(SIS+XS0h+SS2O3+1e-4)
    CS0a = XS0a/(SIS+XS0a+SS2O3+1e-4)
    CS0h = XS0h/(SIS+XS0h+SS2O3+1e-4)
    CTSa = SS2O3/(SIS+XS0a+SS2O3+1e-4)
    CTSh = SS2O3/(SIS+XS0h+SS2O3+1e-4)

    # pH limitation model (Batstone et al., 2002)
    if pH > p.pHUL:
        IpH = 1
    else:
        IpH = np.exp(-3*((pH-p.pHUL)/(p.pHUL-p.pHLL))**2)

    # Rates
    # biological
    rHyd = p.kHyd*XS*IpH
    rAc = p.kMac*ISAc*IFA*IIN*IIP*XPBh*IpH
    rPh = p.kMph*ISS*IFA*IIN*IIP*XPBh*IpH
    rCh = p.kMch*ISS*IFA*IIN*IIP*XPBh*IpH
    rIsa = p.k0a*IISa*IIC*IFA*IIN*IIP*CISa*XPBa*IpH
    rIsh = p.k0h*IISh*IIC*IFA*IIN*IIP*CISh*XPBh*IpH
    rAa = p.kMh2*IH2*IIC*IFA*IIN*IIP*XPBh*IpH
    rDeca = p.kDec*XPBa
    rDech = p.kDec*XPBh
    rSS2O3a = p.k3a*ISS2O3a*IIC*IFA*IIN*IIP*CTSa*XPBa*IpH
    rSS2O3h = p.k3h*ISS2O3h*IIC*IFA*IIN*IIP*CTSh*XPBh*IpH
    rS0a = p.k1a*IXS0a*IIC*IFA*IIN*IIP*CS0a*XPBa*IpH
    rS0h = p.k1h*IXS0h*IIC*IFA*IIN*IIP*CS0h*XPBh*IpH
    r2a = p.k2a*I2a*IIC*IFA*IIN*IIP*CISa*XPBa*IpH
    r2h = p.k2h*I2h*IIC*IFA*IIN*IIP*CISh*XPBh*IpH

    # gas-liq mass transfer
    rTic = p.klaIC*(SCO2aq - 12e3*p.KHic*pCO2/100)
    rTis = p.klaIS*(SH2Saq - 64e3*p.KHis*pH2S/1e6)
    rTn2 = p.klaN2*(SN2 - 14e3*p.KHn2*pN2/100)
    rTh2 = p.klaH2*(SH2 - 16e3*p.KHh2*pH2/100)
    rTch4 = p.klaCH4*(SCH4 - 64e3*p.KHch4*pCH4/100)

    # Gas flow out
    pg = pCH4/100+pCO2/100+pH2S/1e6+pN2/100+pH2/100+pH2O
    deltaP = pg - p.P
    if deltaP < 0:
        Fgasout = 0
    else:
        Fgasout = p.kv*np.sqrt(deltaP)

    # Patching of pressure and gas flow rate out
    PAnMsulfide.pg = pg
    PAnMsulfide.F = Fgasout

    # DAE system equations
    dSS = (SSin-SS)*Fliqin/p.Vl \
        + p.fSsxs*rHyd-rPh-rCh
    dSAc = (SAcin-SAc)*Fliqin/p.Vl \
        + p.fAcxs*rHyd \
        - rAc+(1-p.YPBch)*p.fAcch*rCh
    dSIC = (SICin-SIC)*Fliqin/p.Vl \
        + p.fIcxs*rHyd \
        - p.Cm*(p.YPB0a*rIsa
                + p.YPB0h*rIsh
                + p.YPBa*rAa
                + p.YPB1a*rS0a
                + p.YPB1h*rS0h
                + p.YPB2a*r2a
                + p.YPB2h*r2h
                + p.YPB3a*rSS2O3a
                + p.YPB3h*rSS2O3h) \
        - p.Csi*((0.25-p.YPB3a)*rSS2O3a
                 + (0.25-p.YPB3h)*rSS2O3h
                 + (1-p.YPB1a)*rS0a
                 + (1-p.YPB1h)*rS0h
                 + (0.25-p.YPB0a)*rIsa
                 + (0.25-p.YPB0h)*rIsh
                 + (0.5-p.YPB2a)*r2a
                 + (0.5-p.YPB2h)*r2h) \
        + p.fIcch*rCh - rTic
    dSH2 = (SH2in-SH2)*Fliqin/p.Vl + p.fH2xs*rHyd + (1-p.YPBch)*p.fH2ch*rCh \
        - rAa - rTh2
    dSIS = (SISin-SIS)*Fliqin/p.Vl + p.fIsxs*rHyd - rIsa - rIsh - r2a - r2h \
        - rTis
    dSIN = (SINin-SIN)*Fliqin/p.Vl + p.fInxs*rHyd + p.Nm*(-p.YPBph*rAc \
            - p.YPBph*rPh - p.YPBch*rCh - p.YPB0h*p.fSixpb*rIsh \
            - p.YPB0a*p.fSixpb*rIsa - p.YPB3a*rSS2O3a - p.YPB3h*rSS2O3h \
            - p.YPB1a*rS0a - p.YPB1h*rS0h - p.YPBa*rAa - p.YPB2a*r2a \
            - p.YPB2h*r2h) + (-(0.25-p.YPB3a)*rSS2O3a - (0.25-p.YPB3h)*rSS2O3h \
            - (1-p.YPB1a)*rS0a - (1-p.YPB1h)*rS0h - (0.25-p.YPB0a)*rIsa \
            - (0.25-p.YPB0h)*p.YPB0h*rIsh - (0.5-p.YPB2a)*r2a-(0.5-p.YPB2h)*r2h)*p.Nsi

    dSIP = (SIPin-SIP)*Fliqin/p.Vl + p.fIpxs*rHyd + p.Pm*(-p.YPBph*rAc \
           - p.YPBph*rPh - p.YPBch*rCh - p.YPB0h*p.fSixpb*rIsh \
           - p.YPB0a*p.fSixpb*rIsa - p.YPB3a*rSS2O3a - p.YPB3h*rSS2O3h \
           - p.YPB1a*rS0a - p.YPB1h*rS0h - p.YPBa*rAa \
           - p.YPB2a*r2a-p.YPB2h*r2h) + (-(0.25-p.YPB3a)*rSS2O3a - (0.25-p.YPB3h)*rSS2O3h \
           - (1-p.YPB1a)*rS0a - (1-p.YPB1h)*rS0h - (0.25-p.YPB0a)*rIsa \
           - (0.25-p.YPB0h)*p.YPB0h*rIsh - (0.5-p.YPB2a)*r2a-(0.5-p.YPB2h)*r2h)*p.Psi

    dXS0a = (XS0ain-XS0a)*Fliqin/p.Vl \
        + p.fXS0xs*rHyd \
        + 48/64*rIsa \
        - rS0a \
        + 24/32*rSS2O3a
    dXS0h = (XS0hin-XS0h)*Fliqin/p.Vl + p.fXS0xs*rHyd + 48/64*rIsh - rS0h \
        + 24/32*rSS2O3h
    dSSO4 = (SSO4in-SSO4)*Fliqin/p.Vl + p.fSso4xs*rHyd + 32/48*rS0a \
        + 32/48*rS0h + 16/32*rSS2O3a + 16/32*rSS2O3h
    dSS2O3 = (SS2O3in-SS2O3)*Fliqin/p.Vl + p.fSs2o3xs*rHyd - rSS2O3a \
        - rSS2O3h + 32/64*r2a + 32/64*r2h  # 32/48*r2a
    dSI = (SIin-SI)*Fliqin/p.Vl + p.fSixs*rHyd + (0.25-p.YPB0a)*rIsa \
        + (0.25-p.YPB0h)*rIsh + (0.25-p.YPB3a)*rSS2O3a \
        + (0.25-p.YPB3h)*rSS2O3h + (0.5-p.YPB2a)*r2a + (0.5-p.YPB2h)*r2h \
        + (1-p.YPB1a)*rS0a + (1-p.YPB1h)*rS0h
    dXPBa = (XPBain-XPBa)*Fliqin/p.Vl + p.YPB0a*rIsa + p.YPB2a*r2a \
        + p.YPB1a*rS0a + p.YPB3a*rSS2O3a + p.YPBa*rAa - rDeca
    dXPBh = (XPBhin-XPBh)*Fliqin/p.Vl + p.YPBph*rAc + p.YPBph*rPh \
        + p.YPBch*rCh + p.YPB0h*rIsh + p.YPB2h*r2h + p.YPB1h*rS0h \
        + p.YPB3h*rSS2O3h - rDech
    dXS = (XSin-XS)*Fliqin/p.Vl - rHyd + rDeca + rDech
    dXI = (XIin-XI)*Fliqin/p.Vl + p.fXixs*rHyd
    dSN2 = (SN2in-SN2)*Fliqin/p.Vl - rTn2

    dSICg = 1/p.Vg*(Fgasin*pCO2in - Fgasout*pCO2) \
        + rTic*p.Vl/p.Vg/12e3*p.R*(T+273.15)*100
    dSISg = 1/p.Vg*(Fgasin*pH2Sin - Fgasout*pH2S) \
        + rTis*p.Vl/p.Vg/64e3*p.R*(T+273.15)*1e6
    dpN2 = 1/p.Vg*(Fgasin*pN2in - Fgasout*pN2) \
        + rTn2*p.Vl/p.Vg/14e3*p.R*(T+273.15)*100
    dSH2g = 1/p.Vg*(Fgasin*pH2in - Fgasout*pH2) \
        + rTh2*p.Vl/p.Vg/16e3*p.R*(T+273.15)*100
    dpCH4 = 1/p.Vg*(Fgasin*pCH4in - Fgasout*pCH4) \
        + rTch4*p.Vl/p.Vg/64e3*p.R*(T+273.15)*100
    dSCH4 = (SCH4in-SCH4)*Fliqin/p.Vl - rTch4
    return dSS, dSAc, dSIC, dSH2, dSIS, dSIN, dSIP, dXS0a, dXS0h, dSSO4,\
        dSS2O3, dSI, dXPBa, dXPBh, dXS, dXI, dSN2, dSICg, dSISg, dpN2, dSH2g, dSCH4, dpCH4


def PAnMdynamicInput():
    """
    Function to generate dynamic input for the ODE system in PAnMsulfide.
    Averages of experimental data (run 1) provided.

    Dynamic Input
    -------------
    SSin : np.array
        Soluble solids [mg-COD/L].
    SAcin : np.array
        Soluble acetate [mg-COD/L].
    SICin : np.array
        Soluble inorganic carbon [mg-C/L].
    SH2in : np.array
        Soluble hydrogen [mg-COD/L].
    SISin : np.array
        Soluble inorganic sulfide [mg-COD/L].
    SINin : np.array
        Soluble inorganic nitrogen [mg-N/L].
    SIPin : np.array
        Soluble inorganic phosphorus [mg-P/L].
    XS0ain : np.array
        Particulate elemental sulfur in autotrophs [mg-COD/L].
    XS0hin : np.array
        Particulate elemental sulfur in heterotrophs [mg-COD/L].
    SSO4in : np.array
        Soluble sulfate [mg-S/L].
    SS2O3in : np.array
        Soluble thiosulfate [mg-COD/L].
    SIin : np.array
        Soluble inerts [mg-COD/L].
    XPBain : np.array
        Particulate autotrophic active biomass [mg-COD/L].
    XPBhin : np.array
        Particulate heterotrophic active biomass [mg-COD/L].
    XSin : np.array
        Particulate solids [mg-COD/L].
    XIin : np.array
        Particulate inerts [mg-COD/L].
    SN2in : np.array
        Soluble dinitrogen [mg-N/L].
    pCO2in : np.array
        CO2 partial pressure [vol-%].
    pH2Sin : np.array
        H2S partial pressure [ppmV].
    pN2in : np.array
        N2 partial pressure [vol-%].
    pH2in : np.array
        H2 partial pressure [vol-%].
    Fliqin : np.array
        Liquid flow rate into the system [L/h].
    Fgasin : np.array
        Gas flow rate into the system [L/h].
    T: np.array
        Temperature [degree Celsius]
    pCH4in : np.array
        CH4 partial pressure [vol-%].
    SCH4in : np.array
        Soluble methane [mg-COD/L].

    Returns:
    --------
    u: np.array
        array of all dynamic input values. Units for each component above.
    tu: np.array
        time array for dynamic input [hours].

    """
    input_length = 13022

    SSin = 0*np.ones(input_length)
    SAcin = 0*np.ones(input_length)
    SICin = 377.6*np.ones(input_length)
    SH2in = 0*np.ones(input_length)
    SISin = 0*np.ones(input_length)
    SINin = 408*np.ones(input_length)
    SIPin = 5*np.ones(input_length)
    XS0ain = 0*np.ones(input_length)
    XS0hin = 0*np.ones(input_length)
    SSO4in = 0*np.ones(input_length)
    SS2O3in = 0*np.ones(input_length)
    SIin = 239*np.ones(input_length)
    XPBain = 0*np.ones(input_length)
    XPBhin = 0*np.ones(input_length)
    XSin = 0*np.ones(input_length)
    XIin = 0*np.ones(input_length)
    SN2in = 0*np.ones(input_length)
    pCO2in = 30*np.ones(input_length)
    pH2Sin = 2000*np.ones(input_length)
    pN2in = 0*np.ones(input_length)
    pH2in = 0*np.ones(input_length)
    Fliqin = 0.02088*np.ones(input_length)
    Fgasin = 3.9*np.ones(input_length)  # average provided
    T = 26**np.ones(input_length)  # average provided
    pCH4in = 69.8*np.ones(input_length)
    SCH4in = 0*np.ones(input_length)

    u = SSin, SAcin, SICin, SH2in, SISin, SINin, SIPin, XS0ain, XS0hin, SSO4in,\
        SS2O3in, SIin, XPBain, XPBhin, XSin, XIin, SN2in,\
        pCO2in, pH2Sin, pN2in, pH2in, Fliqin, Fgasin, T, pCH4in, SCH4in
    u = np.transpose(u)

    tu = np.linspace(0, 674.931765, input_length)  # hours

    return u, tu


def PAnMinitialConditions():
    """
    Function to generate initial conditions for the ODE system in PAnMsulfide.

    Returns
    -------
    SS : float
        Soluble solids [mg-COD/L].
    SAc : float
        Soluble acetate [mg-COD/L].
    SIC : float
        Soluble inorganic carbon [mg-C/L].
    SH2 : float
        Soluble hydrogen [mg-COD/L].
    SIS : float
        Soluble inorganic sulfide [mg-COD/L].
    SIN : float
        Soluble inorganic nitrogen [mg-N/L].
    SIP : float
        Soluble inorganic phosphorus [mg-P/L].
    XS0a : float
        Particulate elemental sulfur in autotrophs [mg-COD/L].
    XS0h : float
        Particulate elemental sulfur in heterotrophs [mg-COD/L].
    SSO4 : float
        Soluble sulfate [mg-S/L].
    SS2O3 : float
        Soluble thiosulfate [mg-COD/L].
    SI : float
        Soluble inerts [mg-COD/L].
    XPBa : float
        Particulate autotrophic active biomass [mg-COD/L].
    XPBh : float
        Particulate heterotrophic active biomass [mg-COD/L].
    XS : float
        Particulate solids [mg-COD/L].
    XI : float
        Particulate inerts [mg-COD/L].
    SN2 : float
        Soluble dinitrogen [mg-N/L].
    pCO2 : float
        CO2 partial pressure [vol-%].
    pH2S : float
        H2S partial pressure [ppmV].
    pN2 : float
        N2 partial pressure [vol-%].
    SCH4 : float
        soluble methane [mg-COD/L].
    pCH4 : float
        CH4 partial pressure [vol-%].
    """
    SS = 0
    SAc = 0
    SIC = 377.6
    SH2 = 0
    SIS = 9.6
    SIN = 408
    SIP = 5
    XS0a = 0
    XS0h = 0
    SSO4 = 11.8
    SS2O3 = 9.6
    SI = 239
    XPBa = 0.02
    XPBh = 0.17
    XS = 0
    XI = 0
    SN2 = 0
    pCO2 = 30
    pH2S = 2000
    pN2 = 0
    pH2 = 0
    SCH4 = 0
    pCH4 = 69.8

    return SS, SAc, SIC, SH2, SIS, SIN, SIP, XS0a, XS0h, SSO4, SS2O3, SI, XPBa, XPBh,\
        XS, XI, SN2, pCO2, pH2S, pN2, pH2, SCH4, pCH4


def main():
    """
    Function to solve the ODE system in PAnMsulfide with initial condtions and
    input assignment. Solver method is 'BDF' for stiff systems.
    Returns
    -------
    sol (integrate._ivp.ivp.OdeResult):
        Result of the integration.
        Call result times with:
            result.t
        Call result state variables with:
            result.y
        in the main program.
    """

    y0 = PAnMinitialConditions()
    u, tu = PAnMdynamicInput()

    tspan = (0, 1000)

    solution = solve_ivp(PAnMsulfide, tspan, y0,
                         method='BDF', args=(parameter, u, tu))
    return solution


if __name__ == "__main__":
    result = main()
