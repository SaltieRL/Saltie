import tensorflow as tf

pi = 3.141592653589793
U = 32768.0

tfand = tf.logical_and


class SingleAxisTeacher:

    def __init__(self, batch_size):
        self.batch_size = batch_size
        global zero,zeros3,one
        zero = tf.zeros(self.batch_size, tf.float32)
        zeros3 = [zero,zero,zero]
        one = zero + 1


    def get_output_vector_model(self, state_object):

        steer = pitch = yaw = roll = throttle = boost = jump = powerslide = zero

        player, ball = state_object.gamecars[0], state_object.gameball

        pL,pV,pR = a3(player.Location), a3(player.Velocity), a3(player.Rotation)
        bL,bR,bV = a3(ball.Location), a3(ball.Rotation), a3(ball.Velocity)
        paV,pB = a3(player.AngularVelocity), tf.cast(player.Boost,tf.float32)

        pxv,pyv,pzv = local(pV,zeros3,pR)
        pvd,pva,pvi = spherical(pxv,pyv,pzv)
        iv,rv,av = local(paV,zeros3,pR)

        tx,ty,tz = local(bL,pL,pR)
        txv,tyv,tzv = local(bV,zeros3,pR)
        xv,yv,zv = pxv-txv, pyv-tyv, pzv-tzv

        tL = bL
        tx,ty,bz = local(tL,pL,pR)
        td,ta,ti = spherical(tx,ty,tz)

        x,y,z = local(tL,pL,pR)
        d,a,i = spherical(x,y,z)
        r = pR[2]/U

        # controls
        pitch = regress(-i-iv/15.0)
        # yaw = regress(a-av/13.0)
        # roll = regress(-r+rv/22.0)

        output = [
            throttle,
            steer,
            pitch,
            yaw,
            roll,
            jump,
            tf.cast(boost, tf.float32),
            powerslide
        ]

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
    return tif( tf.abs(value)>R, tf.sign(value)*R, value)

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

def spherical(x,y,z,Urot=True):
    d = tf.sqrt(x*x+y*y+z*z)
    try : i = tf.acos(z/d)
    except: i=0
    a = tf.atan2(y,x)
    if Urot : return d, Range180(a/pi-.5,1), Range180(i/pi-.5,1)
    else : return d,a,i

def cartesian(d,a,i):
    x = d * tf.sin(i) * tf.cos(a)
    y = d * tf.sin(i) * tf.sin(a)
    z = d * tf.cos(i)
    return x,y,z

def d3(A,B=[0,0,0]):
    A,B = a3(A),a3(B)
    return tf.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2+(A[2]-B[2])**2)

def tif(cond, iftrue, iffalse):
  cond = tf.cast(cond, tf.float32)
  return cond*iftrue + (1-cond)*iffalse

def regress(a):
    return tif(abs(a)>.1, tf.sign(a), 10*a)
