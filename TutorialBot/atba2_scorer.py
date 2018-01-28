import tensorflow as tf

pi = 3.141592653589793
U = 32768.0

tfand = tf.logical_and

class TutorialBotOutput:

    def __init__(self, batch_size):
        self.batch_size = batch_size
        global zero,zeros3
        zero = tf.zeros(self.batch_size, tf.float32)
        zeros3 = [zero,zero,zero]

    def get_output_vector(self, values):

        steer = pitch = yaw = roll = throttle = boost = jump = powerslide = zero

        player, ball = values.gamecars[0], values.gameball

        pL,pV,pR = a3(player.Location), a3(player.Velocity), a3(player.Rotation)
        paV,pB = a3(player.AngularVelocity), tf.cast(player.Boost,tf.float32)
        bL,bR,bV = a3(ball.Location), a3(ball.Rotation), a3(ball.Velocity)

        pxv,pyv,pzv = local(pV,zeros3,pR)
        pvd,pva,pvi = spherical(pxv,pyv,pzv)
        iv,rv,av = local(paV,zeros3,pR)

        tx,ty,tz = local(bL,pL,pR)
        txv,tyv,tzv = local(bV,zeros3,pR)
        xv,yv,zv = pxv-txv, pyv-tyv, pzv-tzv

        dT = (.5*tf.abs(ty) + .9*tf.abs(tx) + .34*tf.abs(tz))/1500.0
        tL = predict_ball(bL,bV,dT)

        x,y,z = local(tL,pL,pR)
        d,a,i = spherical(x,y,z)

        # aim
        c = -(player.Team-0.5)*2
        goal = a3([zero,5250*c,zero])
        gx,gy,gz = local(goal,pL,pR)
        gd,ga,gi = spherical(gx,gy,gz)
        ang = Range180(ga-a,1)
        x = x + Range(ang*3000,105)*tf.sign(y)

        d,a,i = spherical(x,y,z)
        r = pR[2]/U

        # controlls
        throttle = regress((y-yv*.23)/900.0)
        steer = regress(a-av/45.0)
        yaw = regress(a-av/13.0)
        pitch = regress(-i-iv/15.0)
        roll = regress(-r+rv/22.0)

        jump = tf.cast( tfand(150<tz, tfand(tz<400 , tfand( tz%300>150, tfand(d<1800, 
                        tf.abs(a-pva)<.03) ) ) ), tf.float32)   

        boost = tf.cast( tfand( tf.abs(a)<.15, tfand( throttle>.5, tf.abs(i)<.25 )), tf.float32)

        powerslide = tf.cast( tfand( throttle*pyv>0.0, tfand( .2<tf.abs(a-av/35.0),
                              tfand( tf.abs(a-av/35.0)<.8, xv>500.0 ) ) ), tf.float32)

        output = [throttle, steer, pitch, yaw, roll, jump, boost, powerslide]

        return output

def a3(V):
    try : a = tf.stack([V.X,V.Y,V.Z])
    except :
        try :a = tf.stack([V.Pitch,V.Yaw,V.Roll])
        except : a = tf.stack([V[0],V[1],V[2]])
    return tf.cast(a,tf.float32)

def Range180(value,pi):
    value = value - tf.abs(value)//(2.0*pi) * (2.0*pi) * tf.sign(value)
    value = value - tf.cast(tf.greater( tf.abs(value), pi),tf.float32) * (2.0*pi) * tf.sign(value)
    return value

def Range(value,R):
    cond = tf.cast(abs(value)>R, tf.float32)
    return cond*tf.sign(value)*R + (1-cond)*value

def rotate2D(x,y,ang):
    x2 = x*tf.cos(ang) - y*tf.sin(ang)
    y2 = y*tf.cos(ang) + x*tf.sin(ang)
    return x2,y2

def local(tL,oL,oR,Urot=True):
    L = tL-oL
    if Urot :
        pitch = oR[0]*pi/U
        yaw = Range180(oR[1]-U/2,U)*pi/U
        roll = oR[2]*pi/U
        R = -tf.stack([pitch,yaw,roll])
    else :
        R = -oR
    x,y = rotate2D(L[0],L[1],R[1])
    y,z = rotate2D(y,L[2],R[0])
    x,z = rotate2D(x,z,R[2])
    return x,y,z

def spherical(x,y,z):
    d = tf.sqrt(x*x+y*y+z*z)
    try : i = tf.acos(z/d)
    except: i=0
    a = tf.atan2(y,x)
    return d, Range180(a-pi/2,pi)/pi, Range180(i-pi/2,pi)/pi

def d3(A,B=[0,0,0]):
    A,B = a3(A),a3(B)
    return tf.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2+(A[2]-B[2])**2)

def regress(a):
    cond = tf.cast(abs(a)> .1, tf.float32)
    return cond*tf.sign(a) + (1-cond)*10*a

def predict_ball(L0,V0,dt):
    r = 0.03
    g = a3([zero,zero,zero-650.0])

    A = g -r*V0
    nL = L0 + V0*dt + .5*A*dt**2
    return nL