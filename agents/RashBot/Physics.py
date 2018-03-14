from Util import *
import math


r, gc = 0.0305, -650            # air resistance, gravity constant
e2, e1, a = .6, .714, .4        # bounce & friction factor, spin inertia thingy
R = 93                          # ball radius
wx, wy, wz = 8200, 10280, 2050  # field dimensions
gx, gz = 1792, 640              # goal dimensions
cR, cR2, cR3 = 520, 260, 190    # ramp radii


def approx_step(L0, V0, dt):

    g = a3([0, 0, gc]) * (d3(V0) > 0)

    A = g - r * V0
    nV = V0 + A * dt
    nL = L0 + V0 * dt + .5 * A * dt**2

    return nL, nV


def local_space(tL, oL, oR):
    L = a3(tL) - a3(oL)
    oR = a2(oR) * pi / 180
    y, z = rotate2D(L[1], L[2], -oR[0])
    x, z = rotate2D(L[0], z, -oR[1])
    return x, y, z


def global_space(L, oL, oR):
    oR = a2(oR) * pi / 180
    tL = a3([0, 0, 0])
    tL[0], tL[2] = rotate2D(L[0], L[2], oR[1])
    tL[1], tL[2] = rotate2D(L[1], tL[2], oR[0])
    tL = a3(tL) + a3(oL)
    return tL


def CollisionFree(L):
    if 242 < L[2] < 1833:
        if abs(L[0]) < 3278:
            if abs(L[1]) < 4722:
                if (abs(L[0]) + abs(L[1])) / 7424 <= 1:
                    return True
    return False


def Collision_R(L):
    x, y, z = L
    cx, cy, cz = wx / 2 - cR, wy / 2 - cR, wz - cR
    cx2, cz2 = wx / 2 - cR2, cR2
    cy3, cz3 = wy / 2 - cR3, cR3

    # Top Ramp X-axis
    if abs(x) > wx / 2 - cR and z > cz and (abs(x) - cx)**2 + (z - cz)**2 > (cR - R)**2:
        a = math.atan2(z - cz, abs(x) - cx) / pi * 180
        return True, [0, (90 + a) * sign(x)]

    # Top Ramp Y-axis
    if abs(y) > cy and z > cz and (abs(y) - cy)**2 + (z - cz)**2 > (cR - R)**2:
        a = math.atan2(z - cz, abs(y) - cy) / pi * 180
        return True, [(90 + a) * sign(y), 0]

    # Bottom Ramp X-axis
    elif abs(x) > cx2 and z < cz2 and (abs(x) - cx2)**2 + (z - cz2)**2 > (cR2 - R)**2:
        a = math.atan2(z - cz2, abs(x) - cx2) / pi * 180
        return True, [0, (90 + a) * sign(x)]

    # Bottom Ramp Y-axis
    elif abs(y) > cy3 and z < cz3 and abs(x) > gx / 2 - R / 2 and (abs(y) - cy3)**2 + (z - cz2)**2 > (cR3 - R)**2:
        a = math.atan2(z - cz2, abs(y) - cy3) / pi * 180
        return True, [(90 + a) * sign(y), 0]

    # Flat 45° Corner
    elif (abs(x) + abs(y) + R) / 8060 >= 1:
        return True, [90 * sign(y), 45 * sign(x)]

    # Floor
    elif z < R:
        return True, [0, 0]

    # Flat Wall X-axis
    elif abs(x) > wx / 2 - R:
        return True, [0, 90 * sign(x)]

    # Flat Wall Y-axis
    elif abs(y) > wy / 2 - R and (abs(x) > gx / 2 - R / 2 or z > gz - R / 2):
        return True, [90 * sign(y), 0]

    # Ceiling
    elif z > wz - R:
        return True, [0, 180]
        # collision bool, bounce angle (pitch, roll)
        # imagine rotating a ground plane

    else:
        return False, [0, 0]


def step(L0, V0, aV0, dt):

    g = a3([0, 0, gc]) * (d3(V0) > 0)
    # don't apply gravity if the ball is floating

    A = g - r * V0
    nV = V0 + A * dt
    nL = L0 + V0 * dt + .5 * A * dt**2

    naV = aV0

    if not CollisionFree(nL):
        Cl = Collision_R(nL)
        if Cl[0] is True:

            # transorforming velocities to local space
            xv, yv, zv = local_space(V0, [0, 0, 0], Cl[1])
            xav, yav, zav = local_space(aV0, [0, 0, 0], Cl[1])

            # bounce angle
            ang = abs(math.atan2(zv, math.sqrt(xv**2 + yv**2))) / pi * 180

            # some more magic numbers
            e = (e1 - .9915) / (29) * ang + .9915

            # limiting e to range [e1, 1]
            if e < e1:
                e = e1

            rolling = 0
            if abs(zv) < 210:
                rolling = 1

            if not rolling:
                # bounce calculations
                xv, yv, zv = (xv + yav * R * a) * e, (yv - xav * R * a) * e, abs(zv) * e2
                xav, yav = -yv / R, xv / R

            # limiting ball spin
            total_av = math.sqrt(xav**2 + yav**2 + zav**2)
            if total_av > 6:
                xav, yav, zav = 6 * xav / total_av, 6 * yav / total_av, 6 * zav / total_av

            # transorforming velocities back to global/world space
            nV = global_space([xv, yv, zv], [0, 0, 0], Cl[1])
            naV = global_space([xav, yav, zav], [0, 0, 0], Cl[1])

            if rolling:
                if d3(nV) > 565:
                    nV = nV * (1 - .6 * dt)
                else:
                    nV = nV * (1 - .2 * dt)

            # redoing the step with the new velocity
            nL = nL + nV * dt + (-r * nV) * dt**2
            if nL[2] > R:
                nL += g * dt**2
            elif rolling:
                nL[2] = R

    # limiting ball speed
    total_v = d3(nV)
    if total_v > 6000:
        nV[0], nV[1], nV[2] = 6 * nV[0] / total_v, 6 * nV[1] / total_v, 6 * nV[2] / total_v

    return nL, nV, naV


def predict_sim(L0, V0, aV0, dt, tps=1 / 60):

    cL0, cV0, caV0 = L0, V0, aV0

    pt = []
    for i in range(int(dt / tps)):
        cL0, cV0, caV0 = step(cL0, cV0, caV0, tps)
        pt.append([cL0, cV0, caV0, (i + 1) * tps])

    # if dt%tps>0:
    cL0, cV0, caV0 = step(cL0, cV0, caV0, dt % tps)
    pt.append([cL0, cV0, caV0, dt])

    return pt
    # returns (Location, Velocity, Angular Velocity, time from the start of the sim)


def time_solve_z(zL, zV, final_zL):

    a = .35*zL*r  - .5*gc
    b = -zV
    c = -zL + final_zL

    return quadratic_pos(a, b, c)
