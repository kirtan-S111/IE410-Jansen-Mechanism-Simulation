import math
import argparse
import numpy as np

def cci(C1, r1, C2, r2, branch):
    dx = C2[0] - C1[0]
    dy = C2[1] - C1[1]
    d = math.hypot(dx, dy)
    if d > r1 + r2: d = r1 + r2 - 1e-9
    if d < abs(r1 - r2): d = abs(r1 - r2) + 1e-9
    a = (r1*r1 - r2*r2 + d*d) / (2*d)
    h_sq = max(0, r1*r1 - a*a)
    h = math.sqrt(h_sq)
    xm = C1[0] + a*dx/d
    ym = C1[1] + a*dy/d
    px = -dy/d
    py = dx/d
    s = 1 if branch == '+' else -1
    return np.array([xm + s*h*px, ym + s*h*py])

def generate_xml(L4_override=None, L8_override=None):
    L = {
        'L1': 11.0, 'L2': 45.0, 'L3': 36.0, 'L4': 33.0,
        'L5': 48.5, 'L6': 41.5, 'L7': 60.5, 'L8': 41.5,
        'L9': 42.0, 'L10': 43.0, 'L11': 26.5, 'L12': 54.5
    }
    
    # Scale to metres
    for k in L: L[k] /= 100.0
    
    if L4_override is not None: L['L4'] = L4_override
    if L8_override is not None: L['L8'] = L8_override

    P0 = np.array([0.0, 0.0])
    P3 = np.array([L['L4'], 0.0])
    P1 = np.array([L['L1'], 0.0])

    P2 = cci(P1, L['L2'],  P3, L['L3'],  '+')
    P5 = cci(P1, L['L7'],  P3, L['L8'],  '-')
    P4 = cci(P2, L['L5'],  P3, L['L6'],  '-')
    P6 = cci(P4, L['L10'], P5, L['L9'],  '-')
    PE = cci(P5, L['L11'], P6, L['L12'], '-')

    def fmt(pt):
        return f"{pt[0]:.6f} 0.0 {pt[1]:.6f}"

    def g(p1, p2, name, color="0.5 0.5 0.5 1"):
        return f'<geom name="{name}" type="capsule" fromto="{fmt(p1)} {fmt(p2)}" size="0.008" rgba="{color}" contype="0" conaffinity="0"/>'
        
    def s(pos, name):
        return f'<site name="{name}" pos="{fmt(pos)}" size="0.01" rgba="1 0 0 1"/>'

    crank_geom = g(np.array([0,0]), P1-P0, "L1", "0.8 0.2 0.2 1")
    crank_site = s(P1-P0, "s_P1")

    link2_geom = g(np.array([0,0]), P2-P1, "L2", "0.2 0.2 0.8 1")
    link2_site = s(P2-P1, "s_P2_link2")

    link7_geom = g(np.array([0,0]), P5-P1, "L7", "0.9 0.5 0.1 1")
    link7_site = s(P5-P1, "s_P5_link7")

    link8_geom = g(np.array([0,0]), P5-P3, "L8", "0.9 0.5 0.1 1")
    link8_site = s(P5-P3, "s_P5_link8")

    c_g1 = g(np.array([0,0]), P2-P3, "L3", "0.1 0.7 0.1 1")
    c_g2 = g(np.array([0,0]), P4-P3, "L6", "0.1 0.7 0.1 1")
    c_g3 = g(P2-P3, P4-P3, "L5", "0.1 0.7 0.1 1")
    c_s2 = s(P2-P3, "s_P2_tri")
    c_s4 = s(P4-P3, "s_P4_tri")

    link10_geom = g(np.array([0,0]), P6-P4, "L10", "0.5 0.2 0.7 1")
    link10_site = s(P6-P4, "s_P6_link10")

    f_g1 = g(np.array([0,0]), P6-P5, "L9", "0.8 0.2 0.8 1")
    f_g2 = g(np.array([0,0]), PE-P5, "L11", "0.8 0.2 0.8 1")
    f_g3 = g(P6-P5, PE-P5, "L12", "0.8 0.2 0.8 1")
    f_s5 = s(np.array([0,0]), "s_P5_foot")
    f_s6 = s(P6-P5, "s_P6_foot")
    f_foot = s(PE-P5, "s_foot")

    xml = f"""<mujoco model="theo_jansen_v2">
    <option gravity="0 0 -9.81" timestep="0.001" iterations="200" tolerance="1e-10"/>
    <visual>
        <global fovy="45"/>
    </visual>
    <worldbody>
        <light pos="0 -2 0" dir="0 1 0" directional="true"/>
        <geom name="ground" type="plane" size="2 2 0.1" pos="0 0 -0.5" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>
        
        <!-- White background plane far away -->
        <geom name="white_bg" type="plane" size="5 5 0.1" pos="0 2 0" axisangle="1 0 0 90" rgba="1 1 1 1" contype="0" conaffinity="0"/>

        <!-- Mounts -->
        <geom type="cylinder" pos="{fmt(P0)}" size="0.015 0.015" axisangle="1 0 0 90" rgba="0 0 0 1" contype="0" conaffinity="0"/>
        <geom type="cylinder" pos="{fmt(P3)}" size="0.015 0.015" axisangle="1 0 0 90" rgba="0 0 0 1" contype="0" conaffinity="0"/>

        <body name="crank" pos="{fmt(P0)}">
            <joint name="joint_crank" type="hinge" pos="0 0 0" axis="0 1 0" damping="0.05" frictionloss="0.0"/>
            {crank_geom}
            {crank_site}

            <body name="link2" pos="{fmt(P1-P0)}">
                <joint name="joint_link2" type="hinge" pos="0 0 0" axis="0 1 0" damping="0.05" frictionloss="0.0"/>
                {link2_geom}
                {link2_site}
            </body>

            <body name="link7" pos="{fmt(P1-P0)}">
                <joint name="joint_link7" type="hinge" pos="0 0 0" axis="0 1 0" damping="0.05" frictionloss="0.0"/>
                {link7_geom}
                {link7_site}
            </body>
        </body>

        <body name="link8" pos="{fmt(P3)}">
            <joint name="joint_link8" type="hinge" pos="0 0 0" axis="0 1 0" damping="0.05" frictionloss="0.0"/>
            {link8_geom}
            {link8_site}

            <body name="foot_tri" pos="{fmt(P5-P3)}">
                <joint name="joint_foot_tri" type="hinge" pos="0 0 0" axis="0 1 0" damping="0.05" frictionloss="0.0"/>
                {f_g1}
                {f_g2}
                {f_g3}
                {f_s5}
                {f_s6}
                {f_foot}
            </body>
        </body>

        <body name="coupler_tri" pos="{fmt(P3)}">
            <joint name="joint_coupler_tri" type="hinge" pos="0 0 0" axis="0 1 0" damping="0.05" frictionloss="0.0"/>
            {c_g1}
            {c_g2}
            {c_g3}
            {c_s2}
            {c_s4}

            <body name="link10" pos="{fmt(P4-P3)}">
                <joint name="joint_link10" type="hinge" pos="0 0 0" axis="0 1 0" damping="0.05" frictionloss="0.0"/>
                {link10_geom}
                {link10_site}
            </body>
        </body>
    </worldbody>

    <equality>
        <connect body1="link2" body2="coupler_tri" anchor="{fmt(P2-P1)}" solref="0.01 1" solimp="0.95 0.99 0.001"/>
        <connect body1="link7" body2="link8" anchor="{fmt(P5-P1)}" solref="0.01 1" solimp="0.95 0.99 0.001"/>
        <connect body1="link10" body2="foot_tri" anchor="{fmt(P6-P4)}" solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </equality>

    <actuator>
        <!-- Drive the crank with a position actuator -->
        <position name="motor_crank" joint="joint_crank" kp="10"/>
    </actuator>
</mujoco>
"""
    with open("Theo Jansen mechanism.xml", "w") as f:
        f.write(xml)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--L4', type=float, default=None)
    parser.add_argument('--L8', type=float, default=None)
    args = parser.parse_args()
    generate_xml(L4_override=args.L4, L8_override=args.L8)
