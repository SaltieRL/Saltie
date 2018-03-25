from Util import *
from Physics import *


def pre_process(s, game):

    s.game = game
    s.player = game.gamecars[s.index]
    s.ball = game.gameball
    s.info = game.gameInfo

    s.time = s.info.TimeSeconds
    s.bH = s.info.bBallHasBeenHit

    s.pL = a3(s.player.Location)
    s.pR = a3(s.player.Rotation)
    s.pV = a3(s.player.Velocity)
    s.paV = a3(s.player.AngularVelocity)
    s.pJ = s.player.bJumped
    s.pdJ = s.player.bDoubleJumped
    s.poG = s.player.bOnGround
    s.pB = s.player.Boost
    s.pS = s.player.bSuperSonic

    s.bL = a3(s.ball.Location)
    s.bV = a3(s.ball.Velocity)
    s.baV = a3(s.ball.AngularVelocity)

    s.bx, s.by, s.bz = local(s.bL, s.pL, s.pR)
    s.bd, s.ba, s.bi = spherical(s.bx, s.by, s.bz)
    s.iv, s.rv, s.av = local(s.paV, z3, s.pR)
    s.pxv, s.pyv, s.pzv = local(s.pV, z3, s.pR)
    s.bxv, s.byv, s.bzv = local(s.bV, z3, s.pR)
    s.pvd, s.pva, s.pvi = spherical(s.pxv, s.pyv, s.pzv)

    s.color = -sign(s.player.Team)

    if not hasattr(s, 'counter'):

        s.counter = -1

        s.throttle = s.steer = s.pitch = s.yaw = s.roll = s.jump = s.boost = 0
        s.powerslide = s.ljump = 0

        s.aT = s.gT = s.sjT = s.djT = s.time

        s.goal = a3([0, 5180 * s.color, 0])
        s.ogoal = a3([0, -5180 * s.color, 0])

        s.a = s.i = s.dT = 0
        s.dodge = s.jumper = s.shoot = 0
        s.brakes = 1
        s.tL = s.bL
        s.djL = 'bL'

        feedback(s)

    if s.poG and not s.lpoG: s.gT = s.time
    if s.lpoG and not s.poG: s.aT = s.time

    s.airtime = s.time - s.aT
    s.gtime = s.time - s.gT
    s.djtime = s.time - s.djT

    if s.lljump and not s.ljump or s.airtime > 0.2: s.sjT = s.ltime

    s.sjtime = s.time - s.sjT  # second jump timer

    if s.poG: s.airtime = s.sjtime = s.djtime = 0
    else: s.gtime = 0

    if s.poG:
        s.jcount = 2
    elif s.pdJ or (s.sjtime > 1.25 and s.pJ):
        s.jcount = 0
    else:
        s.jcount = 1

    if s.jcount == 0 or s.poG: s.dodge = s.jumper = 0

    s.dtime = s.time - s.ltime
    if s.dtime != 0: s.fps = 1 / s.dtime
    else: s.fps = 0

    s.oppIndex = not s.index

    if game.numCars > 2:
        s.oppIndex = -1
        for i in range(game.numCars):
            if game.gamecars[i].Team != s.player.Team:
                if s.oppIndex == -1 or (d3(a3(game.gamecars[i].Location), s.bL) <
                                        d3(a3(game.gamecars[s.oppIndex].Location),
                                           s.bL)):
                    s.oppIndex = i

    s.opp = game.gamecars[s.oppIndex]

    s.oL = a3(s.opp.Location)
    s.oV = a3(s.opp.Velocity)
    s.oR = a3(s.opp.Rotation)


def gather_info(s):

    gy = 5180

    # player info

    s.pdT = Range(d3(s.pL + s.pV / 60, s.bL + s.bV / 60) / 2500, 5)

    s.ptL = step(s.bL, s.bV, s.baV, s.pdT)[0]
    s.pfL = step(s.pL, s.pV, z3, s.pdT)[0]

    s.glinex = line_intersect(([0, gy * s.color], [1, gy * s.color]),
                              ([s.pL[0], s.pL[1]], [s.ptL[0], s.ptL[1]]))[0]

    s.glinez = line_intersect(([0, gy * s.color], [1, gy * s.color]),
                              ([s.pL[2], s.pL[1]], [s.ptL[2], s.ptL[1]]))[0]

    s.oglinex = line_intersect(([0, -gy * s.color], [1, -gy * s.color]),
                               ([s.pL[0], s.pL[1]], [s.ptL[0], s.ptL[1]]))[0]

    s.oglinez = line_intersect(([0, -gy * s.color], [1, -gy * s.color]),
                               ([s.pL[2], s.pL[1]], [s.ptL[2], s.ptL[1]]))[0]

    s.bfd = d3(s.pfL, s.ptL)

    # opponnent info

    s.odT = Range(d3(s.oL + s.oV / 60, s.bL + s.bV / 60) / 2500, 5)

    s.otL = step(s.bL, s.bV, s.baV, s.odT)[0]
    s.ofL = step(s.pL, s.pV, z3, s.odT)[0]

    s.ooglinex = line_intersect(([0, -gy * s.color], [1, -gy * s.color]),
                                ([s.oL[0], s.oL[1]], [s.otL[0], s.otL[1]]))[0]

    s.ooglinez = line_intersect(([0, -gy * s.color], [1, -gy * s.color]),
                                ([s.oL[2], s.oL[1]], [s.otL[2], s.otL[1]]))[0]

    s.obd = d3(s.oL, s.bL)
    s.obfd = d3(s.ofL, s.otL)

    # other

    s.goal = a3([-Range(s.glinex, 550), gy * s.color, 150])

    s.ogoal = a3([Range(s.ooglinex, 900), -gy * s.color,
                  Range(s.ooglinez * .25, 650)])

    s.gaimdx = abs(s.goal[0] - s.glinex)
    s.gaimdz = abs(s.goal[2] - s.glinez)

    s.gx, s.gy, s.gz = local(s.goal, s.pL, s.pR)
    s.gd, s.ga, s.gi = spherical(s.gx, s.gy, s.gz)

    s.ogx, s.ogy, s.ogz = local(s.ogoal, s.pL, s.pR)
    s.ogd, s.oga, s.ogi = spherical(s.ogx, s.ogy, s.ogz)

    s.ogtd = d3(s.ogoal, s.tL)
    s.ogpd = d3(s.ogoal, s.pL)


def feedback(s):

    s.lpoG = s.poG
    s.ltime = s.time
    s.lljump = s.ljump
    s.ljump = s.jump

    s.counter += 1
