"""``robot``: command line to run the controllers in the simulator, regenerate the figures and validate the package.

Subcommand and flag tokens are ASCII-transliterated Turkish (``git`` = go, ``takip`` = follow, ``guvenlik`` = safety, ``dogrulama`` = validation);
the output is Turkish.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys

import numpy as np

from . import __version__


def _fmt_pose(p):
    return f"({p[0]:.2f}, {p[1]:.2f}, {math.degrees(p[2]):.0f} derece)"


def cmd_kinematik(args):
    from . import kinematics as K
    ref = (0.5 / 0.8 * math.sin(2.4), 0.5 / 0.8 * (1 - math.cos(2.4)))
    print("Sabit (v, w) = (0.5 m/s, 0.8 rad/s) ile 3 s sonraki konum hatasi (analitik yay ile karsilastirma):")
    print(f"{'adim sayisi':>12} {'dt (s)':>9} {'Euler':>12} {'orta nokta':>12} {'tam yay':>12}")
    for n in (10, 100, 1000):
        row = []
        for step in (K.integrate_euler, K.integrate_midpoint, K.integrate_exact):
            pose = (0.0, 0.0, 0.0)
            for _ in range(n):
                pose = step(pose, 0.5, 0.8, 3.0 / n)
            row.append(math.hypot(pose[0] - ref[0], pose[1] - ref[1]))
        print(f"{n:>12} {3.0 / n:>9.4f} {row[0]:>12.2e} {row[1]:>12.2e} {row[2]:>12.2e}")
    print("Euler birinci, orta nokta ikinci mertebeden yakinsar; tam yay adim boyundan bagimsiz olarak (yuvarlama hatasi dahilinde) dogrudur.")
    if args.png:
        from . import plots
        plots.plot_integrators(args.png)
        print(f"Grafik yazildi: {args.png}")
    return 0


def _world(name):
    from .sim import World, demo_world
    return World() if name == "bos" else demo_world()


def cmd_git(args):
    from . import scenarios as SC
    start = (args.baslangic[0], args.baslangic[1], math.radians(args.baslangic[2]))
    goal = (args.hedef[0], args.hedef[1], math.radians(args.aci))
    world = _world(args.dunya)
    from .go_to_goal import GoToGoal
    run = SC.run_go_to_goal(start, goal, world, safety=not args.guvenliksiz, controller=GoToGoal(allow_reverse=not args.ileri_only), t_max=args.sure)
    print(f"Baslangic {_fmt_pose(start)}  ->  hedef {_fmt_pose(goal)}   dunya: {args.dunya}, guvenlik: {'kapali' if args.guvenliksiz else 'acik'}")
    if run.collided:
        print(f"SONUC: CARPISMA, {run.duration:.1f} s sonra {_fmt_pose(run.final)} konumunda.")
    elif run.done:
        err = math.hypot(run.final[0] - goal[0], run.final[1] - goal[1])
        length = float(np.sum(np.hypot(*np.diff(run.pose[:, :2], axis=0).T)))
        print(f"SONUC: hedefe ulasti, {run.duration:.1f} s, konum hatasi {err * 100:.1f} cm, yol {length:.2f} m, guvenlik {run.limited_steps} adimda hizi sinirladi.")
    else:
        print(f"SONUC: hedefe ulasamadi (durdu), {run.duration:.1f} s sonra {_fmt_pose(run.final)}; guvenlik {run.limited_steps} adimda hizi sinirladi.")
    if args.png:
        from . import plots
        plots.plot_go_to_goal(run, goal, args.png)
        print(f"Grafik yazildi: {args.png}")
    return 0 if run.done and not run.collided else 1


def cmd_takip(args):
    from . import pure_pursuit as PP
    from . import scenarios as SC
    if args.yol == "cember":
        route, start = PP.circle_path(0, 0, 2.0, turns=1.0), (2.0, 0.0, math.pi / 2)
    elif args.yol == "sekiz":
        route, start = PP.figure_eight_path(), (0.0, 0.0, 0.6)
    else:
        route, start = PP.line_path(0, 0, 8, 0), (0.0, 0.6, 0.0)
    ctl = PP.PurePursuit(lookahead=args.bakis, v_nominal=args.hiz)
    run = SC.run_pure_pursuit(route, start, ideal=args.gecikmesiz, controller=ctl)
    err = SC.cross_track(run, route)
    travelled = float(np.sum(np.hypot(*np.diff(run.pose[:, :2], axis=0).T)))
    print(f"Yol: {args.yol}, uzunluk {PP.path_length(route):.2f} m, bakis mesafesi {args.bakis} m, hiz {args.hiz} m/s, motor gecikmesi {'yok' if args.gecikmesiz else 'var'}")
    print(f"{'bitti' if run.done else 'bitmedi'} {run.duration:.1f} s, katedilen {travelled:.2f} m, en buyuk yanal hata {err.max() * 1000:.1f} mm, ortalama {err.mean() * 1000:.1f} mm")
    if args.png:
        from . import plots
        plots.plot_pure_pursuit(run, route, args.png, title="Pure pursuit on a " + {"cember": "circle", "sekiz": "figure eight", "dogru": "line"}[args.yol])
        print(f"Grafik yazildi: {args.png}")
    return 0 if run.done else 1


def cmd_guvenlik(args):
    from . import scenarios as SC
    without = SC.collision_experiment(args.oda, safety=False, seed=args.tohum, t_max=60.0)
    with_safety = SC.collision_experiment(args.oda, safety=True, seed=args.tohum, t_max=60.0)
    print(f"{args.oda} rastgele oda (8 sutun), rastgele baslangic ve hedef, engel bilmeyen ileri-yonlu hedefe-git denetleyicisi:")
    print(f"  guvenliksiz : {without['collision']:>3} carpisma, {without['reached']:>3} ulasti, {without['stalled']:>3} durdu")
    print(f"  guvenlikli  : {with_safety['collision']:>3} carpisma, {with_safety['reached']:>3} ulasti, {with_safety['stalled']:>3} durdu")
    print("Guvenlik katmani tepkiseldir: carpismayi onler ama engelin etrafindan dolasmaz (durmalarin nedeni budur).")
    if args.png:
        from . import plots
        plots.plot_safety(args.png)
        print(f"Grafik yazildi: {args.png}")
    return 0 if with_safety["collision"] == 0 else 1


def cmd_pid(args):
    from . import scenarios as SC
    plain, guarded = SC.pid_step_response(False), SC.pid_step_response(True)
    print("Doyan girisli birinci derece sistemde PI: kp=2, ki=2, sistem kazanci 2, hedef 1.5 (girisin yalnizca 0.75'i gerekir, ama yukselirken giris 1'de doyar)")
    print(f"  anti-windup yok : asim {plain['overshoot']:.3f}, integral tepe {plain['integral_peak']:.3f}, son deger {plain['final']:.4f}")
    print(f"  kosullu integral: asim {max(guarded['overshoot'], 0):.3f}, integral tepe {guarded['integral_peak']:.3f}, son deger {guarded['final']:.4f}")
    print(f"Kararli durumda integral = u_ss / ki = {guarded['integral_steady']:.3f} olmali; iki durumda da son integral {plain['integral_final']:.3f} / {guarded['integral_final']:.3f}.")
    if args.png:
        from . import plots
        plots.plot_pid(args.png)
        print(f"Grafik yazildi: {args.png}")
    return 0


def cmd_ros(args):
    have = importlib.util.find_spec("rclpy") is not None
    print(f"rclpy: {'bulundu' if have else 'bulunamadi (ROS 2 ortami kaynaklanmamis; cekirdek ve simulator yine de calisir)'}")
    print("Dugumler: python3 -m robot_kontrol.ros.<ad>  (sim_node, mux_node, safety_node, go_to_goal_node, pure_pursuit_node)")
    print("Hepsini birlikte: ros2 launch launch/sim_goto.launch.py   veya   launch/sim_pursuit.launch.py")
    print("Konu akisi: go_to_goal/pure_pursuit -cmd_vel_nav-> mux -cmd_vel_muxed-> safety -cmd_vel-> sim ; teleop -> cmd_vel_teleop ; sim -> odom, scan, collision")
    print("Docker ile (macOS dahil): bash tools/ros_container_test.sh")
    return 0


def cmd_dogrulama(args):
    from . import checks
    results = checks.run_all()
    for c in results:
        print(f"[{'GECTI' if c.passed else 'KALDI'}] {c.name}\n        {c.detail}")
    failed = [c for c in results if not c.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} dogrulama gecti.")
    return 0 if not failed else 1


def build_parser():
    p = argparse.ArgumentParser(prog="robot", description="Basit robot kontrolu: kinematik, PID, hedefe-git, yol takibi, tarama tabanli guvenlik (simulatorde ve ROS 2'de).")
    p.add_argument("--surum", action="version", version=f"robot {__version__}")
    sub = p.add_subparsers(dest="komut", required=True)

    s = sub.add_parser("kinematik", help="integrator hatalarini analitik yayla karsilastir")
    s.add_argument("--png")
    s.set_defaults(func=cmd_kinematik)

    s = sub.add_parser("git", help="bir hedefe git (simulatorde)")
    s.add_argument("--hedef", type=float, nargs=2, metavar=("X", "Y"), required=True)
    s.add_argument("--aci", type=float, default=0.0, help="hedef yonelimi (derece)")
    s.add_argument("--baslangic", type=float, nargs=3, metavar=("X", "Y", "DERECE"), default=[0.0, 0.0, 0.0])
    s.add_argument("--dunya", choices=["bos", "oda"], default="oda")
    s.add_argument("--guvenliksiz", action="store_true", help="tarama guvenligini kapat")
    s.add_argument("--sadece-ileri", action="store_true", dest="ileri_only", help="geri vitesi yasakla")
    s.add_argument("--sure", type=float, default=120.0, help="en fazla benzetim suresi (s)")
    s.add_argument("--png")
    s.set_defaults(func=cmd_git)

    s = sub.add_parser("takip", help="bir yolu pure pursuit ile izle")
    s.add_argument("--yol", choices=["cember", "sekiz", "dogru"], default="cember")
    s.add_argument("--bakis", type=float, default=0.6, help="bakis mesafesi (m)")
    s.add_argument("--hiz", type=float, default=0.3, help="nominal hiz (m/s)")
    s.add_argument("--gecikmesiz", action="store_true", help="motor gecikmesini kaldir (ideal robot)")
    s.add_argument("--png")
    s.set_defaults(func=cmd_takip)

    s = sub.add_parser("guvenlik", help="rastgele odalarda guvenlikli / guvenliksiz karsilastirma")
    s.add_argument("--oda", type=int, default=20)
    s.add_argument("--tohum", type=int, default=100)
    s.add_argument("--png")
    s.set_defaults(func=cmd_guvenlik)

    s = sub.add_parser("pid", help="anti-windup gosterimi")
    s.add_argument("--png")
    s.set_defaults(func=cmd_pid)

    s = sub.add_parser("ros", help="ROS 2 durumu ve calistirma komutlari")
    s.set_defaults(func=cmd_ros)

    s = sub.add_parser("dogrulama", help="tum dogrulama deneylerini calistir")
    s.set_defaults(func=cmd_dogrulama)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
