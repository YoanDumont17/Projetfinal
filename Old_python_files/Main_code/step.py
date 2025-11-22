import pysoem
import ctypes
import time
import threading

# =========================
#  Paramètres utilisateur
# =========================
IFACE_NAME = "enP8p1s0"

# Mécanique axe linéaire
LEAD_MM_PER_REV = 2.0  # [mm/rev] pas de la vis
MICRONS_PER_MM = 1000.0

# Homing
HOMING_METHOD = 25
RETRAIT_COUNTS = 500
PROBE_COUNTS = 50

# PP (Profile Position)
PP_VEL = 100  # [um/s]
PP_ACC = 5000  # [um/s^2]
PP_DEC = 5000  # [um/s^2]
PP_TIMEOUT = 8.0
PRINT_PERIOD = 0.15
ERR_STUCK_WINDOW = 1.0
ERR_STUCK_DROP_MIN = 200  # counts

# Échelle logique ↔ device
SCALE_COUNTS = 1  # Essayez 1 au lieu de 256
phase_offset = 0

# Gestion TR coincé
NUDGE_REL = 50
NUDGE_ABS = 20

# Fenêtre d'acceptation (device)
POSITION_TOL_COUNTS = 50
WAIT_TR_FALL_TIMEOUT = 0.4

# Convention imposée :
#   "+" = retour = s'éloigner de la switch
LOCK_LOGIC_PLUS_IS_AWAY = True

# ===== Unités linéaires (µm) =====
COUNTS_PER_UM = None  # sera calculé après lecture du driver


# =========================
#  Structures PDO
# =========================
class InputPdo(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ('Errorword', ctypes.c_uint16),
        ('statusword', ctypes.c_uint16),
        ('position_actual_value', ctypes.c_int32),
        ('velocity_actual_value', ctypes.c_int32),
        ('followerrorcodevalue', ctypes.c_uint32)
    ]


class OutputPdo(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ('controlword', ctypes.c_uint16),
        ('modes_of_operation', ctypes.c_int8),
        ('target_position', ctypes.c_int32),
        ('target_velocity', ctypes.c_int32),
    ]


modes_of_operation = {
    'Homing mode': 6,
    'Profile position mode': 1,
}

# =========================
#  Globals
# =========================
pd_thread_stop_event = threading.Event()
master = pysoem.Master()
actual_wkc = 0
device = None
outward_sign = +1
HOME_POSITION_OFFSET = 0  # Position device au moment du homing


# =========================
#  Helpers
# =========================
def decode_statusword(sw):
    bits = {
        0: 'Ready to switch on',
        1: 'Switched on',
        2: 'Operation enabled',
        3: 'Fault',
        4: 'Voltage enabled',
        5: 'Quick stop',
        6: 'Switch on disabled',
        10: 'Target reached',
        12: 'Homing attained',
        13: 'Homing error',
    }
    return [name for bit, name in bits.items() if sw & (1 << bit)]


def read_digital_inputs():
    try:
        data = device.sdo_read(0x60FD, 0, 4)
        return int.from_bytes(data, byteorder='little')
    except:
        return None


def read_home_bit():
    """Retourne 1 si switch libre, 0 si contact, -1 si inconnu."""
    di = read_digital_inputs()
    return ((di >> 2) & 1) if di is not None else -1


def set_profile_position_params(vel=PP_VEL, acc=PP_ACC, dec=PP_DEC):
    device.sdo_write(0x6060, 0, bytes(ctypes.c_int8(1)))
    device.sdo_write(0x6081, 0, bytes(ctypes.c_uint32(vel)))
    device.sdo_write(0x6083, 0, bytes(ctypes.c_uint32(acc)))
    device.sdo_write(0x6084, 0, bytes(ctypes.c_uint32(dec)))
    time.sleep(0.02)


def push_output(outp: OutputPdo):
    device.output = bytes(outp)
    master.send_processdata()
    master.receive_processdata(1000)


def cw_base_enable(outp: OutputPdo):
    outp.controlword = 0x000F
    push_output(outp)


def send_pp_front(outp: OutputPdo, *, relative: bool):
    base = 0x000F
    cw = base | (0x0070 if relative else 0x0030)
    outp.controlword = cw
    push_output(outp)
    time.sleep(0.02)
    outp.controlword = 0x000F
    push_output(outp)


def clear_faults_via_sdo():
    print("\n=== EFFACEMENT DES FAULTS (SDO) ===")
    try:
        device.sdo_write(0x2006, 0, bytes(ctypes.c_uint8(0x01)))
        time.sleep(0.2)
        print("  ✓ 0x2006=0x01 envoyé")
    except Exception as e:
        print(f"  ✗ Échec 0x2006: {e}")
    try:
        for v in [0x55, 0x66, 0x77, 0x88, 0x99, 0xAA]:
            device.sdo_write(0x2006, 0, bytes(ctypes.c_uint8(v)))
            time.sleep(0.02)
        print("  ✓ Séquence 0x55..0xAA envoyée")
    except Exception as e:
        print(f"  ✗ Échec séquence: {e}")


def clear_faults_via_pdo():
    print("\n=== EFFACEMENT DES FAULTS (PDO) ===")
    try:
        outp = OutputPdo()
        outp.controlword = 0x0080
        push_output(outp)
        time.sleep(0.15)
        outp.controlword = 0x0000
        push_output(outp)
        print("  ✓ Fault reset envoyé")
    except Exception as e:
        print(f"  ✗ Échec: {e}")


def check_faults():
    try:
        master.send_processdata()
        master.receive_processdata(1000)
        sw = InputPdo.from_buffer_copy(device.input).statusword
        if sw & (1 << 3):
            print(f"⚠️  FAULT PRÉSENT: SW=0x{sw:04X}")
            try:
                err = int.from_bytes(device.sdo_read(0x603F, 0, 2), 'little')
                print(f"   0x603F: 0x{err:04X}")
            except:
                pass
            try:
                dsp = int.from_bytes(device.sdo_read(0x200F, 0, 4), 'little')
                print(f"   DSP 0x200F: 0x{dsp:08X}")
            except:
                pass
            return True
        print(f"✅ Pas de fault - SW=0x{sw:04X}")
        return False
    except Exception as e:
        print(f"Erreur check_faults: {e}")
        return False


def target_reached():
    master.send_processdata()
    master.receive_processdata(1000)
    sw = InputPdo.from_buffer_copy(device.input).statusword
    return bool(sw & (1 << 10))


def wait_tr_fall(timeout=WAIT_TR_FALL_TIMEOUT):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not target_reached():
            return True
        time.sleep(0.02)
    return False


def processdata_thread():
    global actual_wkc
    while not pd_thread_stop_event.is_set():
        master.send_processdata()
        actual_wkc = master.receive_processdata(10000)
        time.sleep(0.01)


# =========================
#  CONFIG SDO au PRE-OP
# =========================
def config_func(slave_pos: int):
    slv = master.slaves[slave_pos]
    print("=== CONFIG HOMING (SDO) ===")
    slv.sdo_write(0x2001, 0, bytes(ctypes.c_uint8(7)))  # X7
    time.sleep(0.02)
    try:
        slv.sdo_write(0x260B, 0, bytes(ctypes.c_uint8(0x05)))
        print("   Limits: 0x260B=0x05 (NO, sans alarme)")
    except:
        try:
            slv.sdo_write(0x220F, 0, bytes(ctypes.c_uint8(0x05)))
            print("   Limits: 0x220F=0x05 (NO, sans alarme)")
        except:
            pass
    time.sleep(0.02)
    slv.sdo_write(0x6098, 0, bytes(ctypes.c_int8(HOMING_METHOD)))
    time.sleep(0.02)
    slv.sdo_write(0x6099, 1, bytes(ctypes.c_uint32(5000)))
    slv.sdo_write(0x6099, 2, bytes(ctypes.c_uint32(500)))
    time.sleep(0.02)
    slv.sdo_write(0x609A, 0, bytes(ctypes.c_uint32(50000)))
    time.sleep(0.02)
    slv.sdo_write(0x607C, 0, bytes(ctypes.c_int32(0)))
    time.sleep(0.02)
    print("✓ Config homing OK.\n")


# =========================
#  PP move
# =========================
def pp_move_absolute(device_target: int, timeout=PP_TIMEOUT):
    set_profile_position_params(PP_VEL, PP_ACC, PP_DEC)

    outp = OutputPdo()
    outp.modes_of_operation = modes_of_operation['Profile position mode']
    outp.target_velocity = 0
    cw_base_enable(outp)

    outp.target_position = int(device_target)
    push_output(outp)
    time.sleep(0.005)
    push_output(outp)

    wait_tr_fall(0.2)

    send_pp_front(outp, relative=False)

    if not wait_tr_fall(0.3):
        master.send_processdata()
        master.receive_processdata(1000)
        pos = InputPdo.from_buffer_copy(device.input).position_actual_value
        pre = pos + (NUDGE_ABS if device_target <= pos else -NUDGE_ABS)
        outp.target_position = int(pre)
        push_output(outp)
        time.sleep(0.005)
        push_output(outp)
        send_pp_front(outp, relative=False)
        wait_tr_fall(0.25)
        outp.target_position = int(device_target)
        push_output(outp)
        time.sleep(0.005)
        push_output(outp)
        send_pp_front(outp, relative=False)
        if not wait_tr_fall(0.3):
            rel = +NUDGE_REL if device_target >= pos else -NUDGE_REL
            outp.target_position = int(rel)
            push_output(outp)
            send_pp_front(outp, relative=True)
            wait_tr_fall(0.25)
            outp.target_position = int(device_target)
            push_output(outp)
            time.sleep(0.005)
            push_output(outp)
            send_pp_front(outp, relative=False)

    print(f"\n=== MOVE ABS vers {device_target} counts ===")
    t0 = time.time()
    last_print = 0.0
    err_window_start = time.time()
    err_at_window_start = None

    while time.time() - t0 < timeout:
        master.send_processdata()
        master.receive_processdata(1000)
        inp = InputPdo.from_buffer_copy(device.input)
        pos = inp.position_actual_value
        tr = bool(inp.statusword & (1 << 10))
        err = pos - device_target

        if err_at_window_start is None:
            err_at_window_start = abs(err)

        now = time.time()
        if now - last_print >= PRINT_PERIOD:
            print(f"\rPos: {pos:10d} | TR: {tr} | err: {err:+d}", end='', flush=True)
            last_print = now

        if tr and abs(err) <= POSITION_TOL_COUNTS:
            print("")
            return True

        if now - err_window_start >= ERR_STUCK_WINDOW:
            drop = err_at_window_start - abs(err)
            if drop < ERR_STUCK_DROP_MIN:
                print("\n⚠️  Err ne diminue plus (bloqué). On abandonne proprement.")
                return False
            err_window_start = now
            err_at_window_start = abs(err)

        time.sleep(0.02)

    print("\n⚠️  TIMEOUT move")
    return False


# =========================
#  Mode interactif RELATIF
# =========================
def interactive_mode_um():
    print("\n" + "=" * 70)
    print("=== MODE POSITIONNEMENT RELATIF (µm depuis position actuelle) ===")
    print("→ Entre un déplacement en µm : +100 (avancer) ou -100 (reculer)")
    print("→ Exemples : +1000 = avancer de 1mm, -500 = reculer de 0.5mm")
    print("Commandes: nombre | 'p' (position) | '0' (retour home) | 'q' (quitter)")
    print("=" * 70)

    while True:
        master.send_processdata()
        master.receive_processdata(1000)
        inp = InputPdo.from_buffer_copy(device.input)
        cur_dev = inp.position_actual_value

        # Affichage position actuelle en µm depuis le home
        if COUNTS_PER_UM:
            cur_um_from_home = (cur_dev - HOME_POSITION_OFFSET) / (outward_sign * COUNTS_PER_UM)
        else:
            cur_um_from_home = float('nan')

        cmd = input(f"Pos: device={cur_dev} (~{cur_um_from_home:.1f} µm depuis home) | Déplacement? ").strip()

        if cmd.lower() == 'q':
            print("Sortie.")
            break

        if cmd.lower() == 'p':
            print(f"Position actuelle: device={cur_dev} (~{cur_um_from_home:.1f} µm depuis home)")
            continue

        if cmd == '0':
            print("→ Retour au point de homing (0)")
            pp_move_absolute(HOME_POSITION_OFFSET)
            continue

        try:
            # Parse le déplacement relatif
            if cmd.startswith('+'):
                delta_um = float(cmd[1:])
            elif cmd.startswith('-'):
                delta_um = -float(cmd[1:])
            else:
                # Si pas de signe, on considère comme positif
                delta_um = float(cmd)

            # Calculer la nouvelle position cible en counts
            delta_counts = round(delta_um * COUNTS_PER_UM)

            # Position cible = position actuelle + delta
            dev_target = cur_dev + (outward_sign * delta_counts)

            print(f"→ Déplacement relatif: {delta_um:+.1f} µm")
            print(f"   Position actuelle: {cur_dev} counts")
            print(f"   Delta: {outward_sign * delta_counts:+d} counts")
            print(f"   Cible: {dev_target} counts")

            pp_move_absolute(dev_target)

        except ValueError:
            print("❌ Entre un nombre (+100, -50, etc.), 'p', '0' ou 'q'.")


# =========================
#  Programme principal
# =========================
def main():
    global master, device, outward_sign, phase_offset, COUNTS_PER_UM, HOME_POSITION_OFFSET

    master.open(IFACE_NAME)
    if master.config_init() <= 0:
        print("❌ No device")
        master.close()
        return

    device = master.slaves[0]
    clear_faults_via_sdo()
    time.sleep(0.2)

    device.config_func = config_func
    master.config_map()

    if master.state_check(pysoem.SAFEOP_STATE, 50_000) != pysoem.SAFEOP_STATE:
        print('❌ SAFEOP_STATE failed')
        master.close()
        return

    master.state = pysoem.OP_STATE
    pd_thread = threading.Thread(target=processdata_thread, daemon=True)
    pd_thread.start()
    master.send_processdata()
    master.receive_processdata(2000)
    master.write_state()
    master.state_check(pysoem.OP_STATE, 5_000_000)
    if master.state != pysoem.OP_STATE:
        print('❌ OP_STATE failed')
        pd_thread_stop_event.set()
        pd_thread.join()
        master.close()
        return

    print("\n" + "=" * 70)
    print("=== VÉRIFICATION DES FAULTS ===")
    print("=" * 70)
    if check_faults():
        clear_faults_via_pdo()
        time.sleep(0.3)
        if check_faults():
            print("❌ Fault persistant, abandon.")
            return

    # Enable (Shutdown → Switch On → Enable Operation)
    outp = OutputPdo()
    outp.modes_of_operation = modes_of_operation['Homing mode']
    outp.controlword = 0x0006
    push_output(outp)
    time.sleep(0.2)
    outp.controlword = 0x0007
    push_output(outp)
    time.sleep(0.2)
    outp.controlword = 0x000F
    push_output(outp)
    time.sleep(0.3)

    # ===== HOMING (driver) =====
    print("\n" + "=" * 70)
    print("=== HOMING (driver) : instrumentation ===")
    print("=" * 70)
    outp.modes_of_operation = modes_of_operation['Homing mode']
    push_output(outp)
    outp.controlword = 0x003F
    push_output(outp)
    time.sleep(0.1)
    outp.controlword = 0x000F
    push_output(outp)

    t0 = time.time()
    homed_pos = None

    # Lecture steps/rev pour établir COUNTS_PER_UM
    try:
        b = device.sdo_read(0x2604, 0, 4)
        steps_per_rev = int.from_bytes(b, "little", signed=False)
        print(f"→ Steps/rev du driver : {steps_per_rev}")
    except Exception:
        steps_per_rev = None
        print("⚠️ Impossible de lire steps/rev, utilisation de 400 par défaut")

    if steps_per_rev and steps_per_rev > 0:
        counts_per_rev = steps_per_rev * SCALE_COUNTS
    else:
        # fallback pour NEMA 0,9°/pas
        counts_per_rev = 400 * SCALE_COUNTS

    # Conversion linéaire (µm)
    if LEAD_MM_PER_REV <= 0:
        raise ValueError("LEAD_MM_PER_REV doit être > 0 (mm par tour).")
    COUNTS_PER_UM = counts_per_rev / (LEAD_MM_PER_REV * MICRONS_PER_MM)
    print(f"→ Mapping linéaire: COUNTS_PER_UM = {COUNTS_PER_UM:.6f} counts/µm")
    print(f"  (steps/rev={steps_per_rev or 400}, SCALE={SCALE_COUNTS}, lead={LEAD_MM_PER_REV} mm/rev)")

    # Attente du homing
    while time.time() - t0 < 45:
        master.send_processdata()
        master.receive_processdata(1000)
        inp = InputPdo.from_buffer_copy(device.input)
        sw = inp.statusword
        pos = inp.position_actual_value
        if (sw & (1 << 12)) and (sw & (1 << 10)):
            print(f"✅ HOMING ATTAINED @ pos={pos}  (SW=0x{sw:04X})")
            homed_pos = pos
            HOME_POSITION_OFFSET = pos  # Mémoriser la position de référence
            print(f"→ Position de référence (home) : {HOME_POSITION_OFFSET} counts")
            break
        time.sleep(0.02)

    if homed_pos is None:
        print("❌ Homing timeout")
        return

    # Détection du sens + (s'éloigner de la switch)
    set_profile_position_params(vel=5000, acc=20000, dec=20000)
    outp_pp = OutputPdo()
    outp_pp.modes_of_operation = modes_of_operation['Profile position mode']
    cw_base_enable(outp_pp)

    # Test +PROBE
    outp_pp.target_position = homed_pos + PROBE_COUNTS
    push_output(outp_pp)
    time.sleep(0.005)
    push_output(outp_pp)
    wait_tr_fall(0.25)
    send_pp_front(outp_pp, relative=False)
    time.sleep(0.2)
    plus_ok = read_home_bit()

    if plus_ok == 1:
        detected_away_sign = +1
    else:
        # retour et test -
        outp_pp.target_position = homed_pos
        push_output(outp_pp)
        time.sleep(0.005)
        push_output(outp_pp)
        wait_tr_fall(0.25)
        send_pp_front(outp_pp, relative=False)
        time.sleep(0.2)

        outp_pp.target_position = homed_pos - PROBE_COUNTS
        push_output(outp_pp)
        time.sleep(0.005)
        push_output(outp_pp)
        wait_tr_fall(0.25)
        send_pp_front(outp_pp, relative=False)
        time.sleep(0.2)
        minus_ok = read_home_bit()
        detected_away_sign = -1 if minus_ok == 1 else +1

    outward_sign = detected_away_sign if LOCK_LOGIC_PLUS_IS_AWAY else detected_away_sign
    print(f"→ Sens imposé: '+' (logique) = s'éloigner ; outward_sign = {outward_sign:+d}")

    # Dégagement de la switch
    outp_pp.target_position = homed_pos + outward_sign * RETRAIT_COUNTS
    push_output(outp_pp)
    time.sleep(0.005)
    push_output(outp_pp)
    wait_tr_fall(0.25)
    send_pp_front(outp_pp, relative=False)
    time.sleep(0.25)

    # Calage de phase
    master.send_processdata()
    master.receive_processdata(1000)
    cur_after = InputPdo.from_buffer_copy(device.input).position_actual_value
    phase_offset = cur_after % SCALE_COUNTS
    print(f"→ phase_offset = {phase_offset} (pos % {SCALE_COUNTS})")

    # État capteur
    print(f"État capteur après dégagement: {read_home_bit()} (1=relâché)")

    # === MODE INTERACTIF RELATIF ===
    try:
        interactive_mode_um()
    except KeyboardInterrupt:
        print("\n🛑 Interruption utilisateur")

    # Arrêt propre
    print("\nDésactivation...")
    outp = OutputPdo()
    outp.controlword = 0x0006
    push_output(outp)
    time.sleep(0.1)
    device.output = bytes(len(device.output))
    master.send_processdata()
    master.receive_processdata(1000)
    pd_thread_stop_event.set()
    pd_thread.join()
    master.state = pysoem.PREOP_STATE
    master.write_state()
    master.close()
    print("✓ Arrêt propre")


if __name__ == "__main__":
    main()