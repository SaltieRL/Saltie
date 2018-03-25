from Util import *
from Physics import *


def strategy(s):

    s.aerialing = not s.poG and s.pL[2] > 150 and s.airtime > .25
    s.kickoff = not s.bH and d3(s.bL) < 99
    s.offense = s.ogtd + 70 > s.ogpd

    s.flip = True
    s.wavedash = True
    s.tojump = True

    if s.kickoff:
        KickoffChase(s)

    else:
        ChaseBallBias(s)

        if s.poG:
            cbL = closest_boost(s, s.tL * .75 + (s.pL + s.pV * s.dT) * .25)
            cbdT = Range(d3(s.pL, cbL) / 2400, 3)

            if s.pB < 33 and cbdT < s.dT * .5:
                GoTo(s, cbL, brakes=0)
                s.pB = 0

    s.r = s.pR[2] / U


def ChaseBallBias(s):

    s.shoot = True

    s.xv, s.yv, s.zv = s.pxv - s.bxv, s.pyv - s.byv, s.pzv - s.bzv
    s.vd, s.va, s.vi = spherical(s.xv, s.yv, s.zv)
    s.vd2 = d2([s.xv, s.yv])

    # dt search
    s.dT = Range(d3(s.pL, s.bL) / 1850, 3.5)
    tps = Range(50 / (s.dT + 1 / 90), 90)
    prediction = predict_sim(s.bL, s.bV, s.baV, s.dT, 1 / tps)

    intercept_state = prediction[0]

    initial_i = i = int(d3(s.pL, s.bL) / 3000 * tps)

    while i < len(prediction):
        state = prediction[i]
        if i == initial_i or ((d3(state[0], step(s.pL, s.pV, z3, state[3])[0]) <
                               intercept_state_distance)):
            intercept_state = state
            intercept_state_distance = d3(state[0],
                                          step(s.pL, s.pV, z3, state[3])[0])
        if intercept_state_distance < 40:
            break
        i += 1

    s.tL, s.tV, s.taV, s.dT = intercept_state

    s.tx, s.ty, s.tz = local(s.tL, s.pL, s.pR)
    s.td, s.ta, s.ti = spherical(s.tx, s.ty, s.tz)
    s.td2 = d2([s.tx, s.ty])

    if s.aerialing:
        s.tL = s.tL - step(z3, s.pV, z3, s.dT)[0]

    if s.pL[2] > 50 and s.poG and (s.tL[2] < s.tz or s.tz > 450 + s.pB * 9):
        s.tL[2] = 93

    s.tx, s.ty, s.tz = local(s.tL, s.pL, s.pR)
    s.td, s.ta, s.ti = spherical(s.tx, s.ty, s.tz)

    aim(s, 70 + s.poG * 36, 1)

    s.brakes = (s.poG and (abs(s.z) > 110 or abs(s.a) > .1)) or not s.tojump


def aim(s, radius, turning_circle=False):

    togL = s.ogoal - s.tL
    tgL = s.goal - s.tL
    tpL = s.pL + s.pV * s.dT / 2 - s.tL

    s.gtL = -tgL
    s.gpL = s.pL - s.goal

    s.gtd, s.gta, s.gti = spherical(*s.gtL, 0)
    s.gpd, s.gpa, s.gpi = spherical(*s.gpL, 0)

    s.tpd, s.tpa, s.tpi = spherical(*tpL, 0)
    s.tgd, s.tga, s.tgi = spherical(*tgL, 0)
    s.togd, s.toga, s.togi = spherical(*togL, 0)

    s.tga = Range180(s.tga + pi, pi)
    s.tgi = Range180(pi - s.tgi, pi)

    if turning_circle and not s.offense:
        radius += Range(Range(ang_dif(s.tga, s.tpa, pi) / pi, .7) * 1.3 *
                        pos(abs(s.ty - s.pyv * s.dT / 2) / 2 - 699) * s.poG * (abs(s.tz) < 150), 599) * .8

    if s.offense:

        tga = mid_ang(s.tpa, s.tga)
        tgi = mid_ang(s.tpi, s.tgi)

        s.tL = cartesian(radius, tga, tgi) + s.tL

    else:

        toga = mid_ang(s.tpa, s.toga)
        togi = mid_ang(s.tpi, s.togi)

        s.tL = cartesian(radius, toga, togi) + s.tL

    s.x, s.y, s.z = local(s.tL, s.pL, s.pR)
    s.d, s.a, s.i = spherical(s.x, s.y, s.z)

    if turning_circle:
        Dwellers_mess(s)

    s.d, s.a, s.i = spherical(s.x, s.y, s.z)

    s.d2 = d2([s.x, s.y])

    if not s.aerialing:
        s.d2pv = d2([s.x - s.pxv * (s.dT + .04), s.y - s.pyv * (s.dT + .04)])
    else:
        s.d2pv = s.d2


def KickoffChase(s):

    s.shoot = True

    s.xv = s.pxv - s.bxv
    s.yv = s.pyv - s.byv
    s.zv = s.pzv - s.bzv
    s.vd, s.va, s.vi = spherical(s.xv, s.yv, s.zv)
    s.vd2 = d2([s.xv, s.yv])

    s.dT = .15
    if s.obd > 1099 and s.opp.bOnGround:
        s.dT = -0.042

    s.tL = s.bL

    if abs(s.pL[0]) > 999:
        s.tL[1] -= Range(abs(s.pL[0]), 999) / 5 * s.color

    s.tx, s.ty, s.tz = local(s.tL, s.pL, s.pR)
    s.td, s.ta, s.ti = spherical(s.tx, s.ty, s.tz)

    aim(s, 90)

    s.brakes = False


def closest_boost(s, tL):

    sdist = 0

    for i in range(34):
        bL = a3(s.game.gameBoosts[i].Location)
        Ac = s.game.gameBoosts[i].bActive
        dist = d3(bL, tL)
        if dist < sdist and Ac or sdist == 0:
            sdist = dist
            L = bL

    return L


def GoTo(s, tL, brakes=True, shoot=False):

    s.brakes = brakes
    s.shoot = shoot

    s.tL = a3(tL)
    tL[2] = 50

    s.xv = s.pxv
    s.yv = s.pyv
    s.zv = s.pzv
    s.vd, s.va, s.vi = s.pvd, s.pva, s.pvi
    s.vd2 = d2([s.xv, s.yv])

    s.tx, s.ty, s.tz = local(s.tL, s.pL, s.pR)

    s.dT = Range(d3(s.pL + s.pV / 5, s.tL) / 2800, 5)

    if s.aerialing:
        s.tL = s.tL - approx_step(z3, s.pV, s.dT)[0]

    s.tx, s.ty, s.tz = local(s.tL, s.pL, s.pR)
    s.td, s.ta, s.ti = spherical(s.tx, s.ty, s.tz)

    s.x, s.y, s.z = s.tx, s.ty, s.tz
    s.d, s.a, s.i = s.td, s.ta, s.ti

    s.d2pv = d2([s.x - s.pxv * s.dT, s.y - s.pyv * s.dT])
    s.d2 = d2([s.x, s.y])


def Dwellers_mess(s):

    s.d2pv = d2([s.tx - s.pxv * s.dT, s.ty - s.pyv * s.dT])
    s.d2 = d2([s.tx, s.ty])

    x = s.x

    if s.offense:
        wa = Range180(Range180(s.ga - s.a, 1) + Range180(s.av / 50, 1), 1)
        s.x += Range(curve3(mnofst(wa, 0.05)) * mnofst(s.ty - s.pyv * .9 * s.dT, 299),
                     499) * (s.z < 140) * s.poG / 2 * (abs(s.gaimdx) > 100)

        s.x = .1 * x + .9 * s.x
