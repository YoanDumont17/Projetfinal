import pysoem   # Bibliothèque pour la communication EtherCAT
import ctypes   # Bibliothèque pour la manipulation de structures C
import time     # Bibliothèque pour la gestion du temps
import threading  # Bibliothèque pour la gestion des threads (threads = processus légers)

class InputPdo(ctypes.Structure):
    ''''Structure représentant les données d'entrée PDO du moteur. 
    Un PDO (Process Data Object) est un format de données utilisé dans les systèmes de communication industrielle comme EtherCAT pour échanger des informations entre les dispositifs.'''

    _pack_ = 1      # Alignement des données à 1 octet
    _fields_ = [
        ('Errorword', ctypes.c_uint16),         # Code d'erreur du moteur sous forme d'entier non signé 16 bits, c'est-à-dire un nombre entier positif
        ('statusword', ctypes.c_uint16),        # Mot de statut du moteur sous forme d'entier non signé 16 bits
        ('position_actual_value', ctypes.c_int32),  # Position actuelle du moteur sous forme d'entier signé 32 bits, c'est-à-dire un nombre entier pouvant être positif ou négatif
        ('velocity_actual_value', ctypes.c_int32),  # Vitesse actuelle du moteur sous forme d'entier signé 32 bits
        ('followerrorcodevalue', ctypes.c_uint32)   # Code d'erreur de suivi sous forme d'entier non signé 32 bits
    ]

class OutputPdo(ctypes.Structure):
    '''Structure représentant les données de sortie PDO du moteur.'''
    _pack_ = 1
    _fields_ = [
        ('controlword', ctypes.c_uint16),      # Mot de contrôle du moteur sous forme d'entier non signé 16 bits
        ('modes_of_operation', ctypes.c_int8),  # Mode de fonctionnement du moteur sous forme d'entier signé 8 bits
        ('target_position', ctypes.c_int32),    # Position cible du moteur sous forme d'entier signé 32 bits
        ('target_velocity', ctypes.c_int32),    # Vitesse cible du moteur sous forme d'entier signé 32 bits
    ]


# Dictionnaire des modes de fonctionnement, on choisit le mode Homing ou Profile position qui sont respectivement les modes 6 et 1 (voir documentation du moteur)
# Provenant de la 
modes_of_operation = {
    'Homing mode': 6,
    'Profile position mode': 1,
}

class Motor_controller:
    '''Classe pour contrôler un moteur via EtherCAT.'''

    def __init__(self):
        '''Initialisation des paramètres du contrôleur de moteur.'''
        self.IFACE_NAME = "enP8p1s0"        # Nom de l'interface réseau utilisée pour la communication EtherCAT du Jetson Nano

        self.LEAD_MM_PER_REV = 2.0          # Pas de la vis en mm par révolution
        self.MICRONS_PER_MM = 1000.0        # Microns par mm

        self.HOMING_METHOD = 25             # Méthode de homing (voir documentation du moteur)
        self.RETRAIT_COUNTS = 500           # Retrait après homing en counts
        self.PROBE_COUNTS = 50              # Distance de sondage en counts

        self.PP_VEL = 100    
        self.PP_ACC = 5000
        self.PP_DEC = 5000

        self.PP_TIMEOUT = 8.0
        self.PRINT_PERIOD = 0.15
        self.ERR_STUCK_WINDOW = 1.0
        self.ERR_STUCK_DROP_MIN = 200
        self.SCALE_COUNTS = 256
        self.phase_offset = 0
        self.NUDGE_REL = 50
        self.NUDGE_ABS = 20
        self.POSITION_TOL_COUNTS = 50
        self.WAIT_TR_FALL_TIMEOUT = 0.4
        self.MAX_POS_UM = 200_000_000
        self.MIN_NEG_UM = -2_000_000
        self.SAFE_BUFFER_UM = 4 * self.MICRONS_PER_MM
        self.LOCK_LOGIC_PLUS_IS_AWAY = True
        self.COUNTS_PER_UM = None
        self.pd_thread_stop_event = threading.Event()
        self.master = pysoem.Master()
        self.actual_wkc = 0
        self.device = None
        self.outward_sign = +1
        self.HOME_POSITION_OFFSET = 0

    def decode_statusword(self, sw):
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

    def read_digital_inputs(self):
        try:
            data = self.device.sdo_read(0x60FD, 0, 4)
            return int.from_bytes(data, byteorder='little')
        except:
            return None

    def read_home_bit(self):
        di = self.read_digital_inputs()
        return ((di >> 2) & 1) if di is not None else -1

    def set_profile_position_params(self, vel=None, acc=None, dec=None):
        if vel is None:
            vel = self.PP_VEL
        if acc is None:
            acc = self.PP_ACC
        if dec is None:
            dec = self.PP_DEC
        self.device.sdo_write(0x6060, 0, bytes(ctypes.c_int8(1)))
        self.device.sdo_write(0x6081, 0, bytes(ctypes.c_uint32(vel)))
        self.device.sdo_write(0x6083, 0, bytes(ctypes.c_uint32(acc)))
        self.device.sdo_write(0x6084, 0, bytes(ctypes.c_uint32(dec)))
        time.sleep(0.02)

    def push_output(self, outp):
        self.device.output = bytes(outp)
        self.master.send_processdata()
        self.master.receive_processdata(1000)

    def cw_base_enable(self, outp):
        outp.controlword = 0x000F
        self.push_output(outp)

    def send_pp_front(self, outp, relative):
        base = 0x000F
        cw = base | (0x0070 if relative else 0x0030)
        outp.controlword = cw
        self.push_output(outp)
        time.sleep(0.02)
        outp.controlword = 0x000F
        self.push_output(outp)

    def clear_faults_via_sdo(self):
        print("\n=== EFFACEMENT DES FAULTS (SDO) ===")
        try:
            self.device.sdo_write(0x2006, 0, bytes(ctypes.c_uint8(0x01)))
            time.sleep(0.2)
            print("  ✓ 0x2006=0x01 envoyé")
        except Exception as e:
            print(f"  ✗ Échec 0x2006: {e}")
        try:
            for v in [0x55, 0x66, 0x77, 0x88, 0x99, 0xAA]:
                self.device.sdo_write(0x2006, 0, bytes(ctypes.c_uint8(v)))
                time.sleep(0.02)
            print("  ✓ Séquence 0x55..0xAA envoyée")
        except Exception as e:
            print(f"  ✗ Échec séquence: {e}")

    def clear_faults_via_pdo(self):
        print("\n=== EFFACEMENT DES FAULTS (PDO) ===")
        try:
            outp = OutputPdo()
            outp.controlword = 0x0080
            self.push_output(outp)
            time.sleep(0.15)
            outp.controlword = 0x0000
            self.push_output(outp)
            print("  ✓ Fault reset envoyé")
        except Exception as e:
            print(f"  ✗ Échec: {e}")

    def check_faults(self):
        try:
            self.master.send_processdata()
            self.master.receive_processdata(1000)
            sw = InputPdo.from_buffer_copy(self.device.input).statusword
            if sw & (1 << 3):
                print(f"⚠️  FAULT PRÉSENT: SW=0x{sw:04X}")
                try:
                    err = int.from_bytes(self.device.sdo_read(0x603F, 0, 2), 'little')
                    print(f"   0x603F: 0x{err:04X}")
                except:
                    pass
                try:
                    dsp = int.from_bytes(self.device.sdo_read(0x200F, 0, 4), 'little')
                    print(f"   DSP 0x200F: 0x{dsp:08X}")
                except:
                    pass
                return True
            print(f"✅ Pas de fault - SW=0x{sw:04X}")
            return False
        except Exception as e:
            print(f"Erreur check_faults: {e}")
            return False

    def target_reached(self):
        self.master.send_processdata()
        self.master.receive_processdata(1000)
        sw = InputPdo.from_buffer_copy(self.device.input).statusword
        return bool(sw & (1 << 10))

    def wait_tr_fall(self, timeout=None):
        if timeout is None:
            timeout = self.WAIT_TR_FALL_TIMEOUT
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not self.target_reached():
                return True
            time.sleep(0.02)
        return False

    def processdata_thread(self):
        while not self.pd_thread_stop_event.is_set():
            self.master.send_processdata()
            self.actual_wkc = self.master.receive_processdata(10000)
            time.sleep(0.01)

    def config_func(self, slave_pos):
        slv = self.master.slaves[slave_pos]
        print("=== CONFIG HOMING (SDO) ===")
        slv.sdo_write(0x2001, 0, bytes(ctypes.c_uint8(7)))
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
        slv.sdo_write(0x6098, 0, bytes(ctypes.c_int8(self.HOMING_METHOD)))
        time.sleep(0.02)
        slv.sdo_write(0x6099, 1, bytes(ctypes.c_uint32(5000)))
        slv.sdo_write(0x6099, 2, bytes(ctypes.c_uint32(500)))
        time.sleep(0.02)
        slv.sdo_write(0x609A, 0, bytes(ctypes.c_uint32(50000)))
        time.sleep(0.02)
        slv.sdo_write(0x607C, 0, bytes(ctypes.c_int32(0)))
        time.sleep(0.02)
        print("✓ Config homing OK.\n")

    def clamp_um(self, x_um):
        x_um = min(x_um, self.MAX_POS_UM)
        min_allowed = -int(self.SAFE_BUFFER_UM)
        x_um = max(x_um, max(self.MIN_NEG_UM, min_allowed))
        return x_um

    def um_to_device_counts(self, x_um):
        if self.COUNTS_PER_UM is None:
            raise RuntimeError("COUNTS_PER_UM non initialisé (LEAD_MM_PER_REV ? 0x2604 ?)")
        x_um = self.clamp_um(int(x_um))
        counts_abs = round(x_um * self.COUNTS_PER_UM)
        print(f"DEBUG Conversion:")
        print(f"  - Consigne : {x_um} µm")
        print(f"  - COUNTS_PER_UM : {self.COUNTS_PER_UM:.6f}")
        print(f"  - Counts calculés : {counts_abs}")
        print(f"  - Counts envoyés : {self.phase_offset + self.outward_sign * counts_abs}")
        return self.phase_offset + self.outward_sign * counts_abs

    def pp_move_absolute(self, device_target, timeout=None, monitor_home_when_negative=False):
        if timeout is None:
            timeout = self.PP_TIMEOUT
        self.set_profile_position_params()
        outp = OutputPdo()
        outp.modes_of_operation = modes_of_operation['Profile position mode']
        outp.target_velocity = 0
        self.cw_base_enable(outp)
        outp.target_position = int(device_target)
        self.push_output(outp)
        time.sleep(0.005)
        self.push_output(outp)
        self.wait_tr_fall(0.2)
        self.send_pp_front(outp, relative=False)
        if not self.wait_tr_fall(0.3):
            self.master.send_processdata()
            self.master.receive_processdata(1000)
            pos = InputPdo.from_buffer_copy(self.device.input).position_actual_value
            pre = pos + (self.NUDGE_ABS if device_target <= pos else -self.NUDGE_ABS)
            outp.target_position = int(pre)
            self.push_output(outp)
            time.sleep(0.005)
            self.push_output(outp)
            self.send_pp_front(outp, relative=False)
            self.wait_tr_fall(0.25)
            outp.target_position = int(device_target)
            self.push_output(outp)
            time.sleep(0.005)
            self.push_output(outp)
            self.send_pp_front(outp, relative=False)
            if not self.wait_tr_fall(0.3):
                rel = +self.NUDGE_REL if device_target >= pos else -self.NUDGE_REL
                outp.target_position = int(rel)
                self.push_output(outp)
                self.send_pp_front(outp, relative=True)
                self.wait_tr_fall(0.25)
                outp.target_position = int(device_target)
                self.push_output(outp)
                time.sleep(0.005)
                self.push_output(outp)
                self.send_pp_front(outp, relative=False)
        print(f"\n=== MOVE ABS {device_target} ===")
        t0 = time.time()
        last_print = 0.0
        err_window_start = time.time()
        err_at_window_start = None
        while time.time() - t0 < timeout:
            self.master.send_processdata()
            self.master.receive_processdata(1000)
            inp = InputPdo.from_buffer_copy(self.device.input)
            pos = inp.position_actual_value
            tr = bool(inp.statusword & (1 << 10))
            err = pos - device_target
            if err_at_window_start is None:
                err_at_window_start = abs(err)
            now = time.time()
            if now - last_print >= self.PRINT_PERIOD:
                print(f"\rPos: {pos:10d} | TR: {tr} | err: {err:+d}", end='', flush=True)
                last_print = now
            if tr and abs(err) <= self.POSITION_TOL_COUNTS:
                print("")
                return True
            if now - err_window_start >= self.ERR_STUCK_WINDOW:
                drop = err_at_window_start - abs(err)
                if drop < self.ERR_STUCK_DROP_MIN:
                    print("\n⚠️  Err ne diminue plus (bloqué). On abandonne proprement.")
                    return False
                err_window_start = now
                err_at_window_start = abs(err)
            time.sleep(0.02)
        print("\n⚠️  TIMEOUT move")
        return False

    def move_to_position_um(self, x_um_abs, vel=None, acc=None, dec=None):
        self.set_profile_position_params(vel, acc, dec)
        dev_tgt = self.um_to_device_counts(x_um_abs)
        monitor = (x_um_abs < 0)
        return self.pp_move_absolute(dev_tgt, monitor_home_when_negative=monitor)

    def move_relative_um(self, delta_um, vel=None, acc=None, dec=None):
        self.master.send_processdata()
        self.master.receive_processdata(1000)
        inp = InputPdo.from_buffer_copy(self.device.input)
        cur_dev = inp.position_actual_value
        if self.COUNTS_PER_UM is None:
            raise RuntimeError("COUNTS_PER_UM non initialisé")
        delta_counts = round(delta_um * self.COUNTS_PER_UM)
        dev_target = cur_dev + (self.outward_sign * delta_counts)
        print(f"→ Déplacement relatif: {delta_um:+.1f} µm")
        print(f"   Position actuelle: {cur_dev} counts")
        print(f"   Delta: {self.outward_sign * delta_counts:+d} counts")
        print(f"   Cible: {dev_target} counts")
        self.set_profile_position_params(vel, acc, dec)
        return self.pp_move_absolute(dev_target)

    def perform_sequence(self, sequence):
        """
        Perform a sequence of relative movements.
        sequence: list of dicts, each with 'delta_um', 'vel' (optional), 'acc' (optional), 'dec' (optional)
        """
        for move in sequence:
            delta_um = move['delta_um']
            vel = move.get('vel')
            acc = move.get('acc')
            dec = move.get('dec')
            success = self.move_relative_um(delta_um, vel, acc, dec)
            if not success:
                print(f"Échec du mouvement relatif de {delta_um} µm")
                return False
        return True

    def interactive_mode_um(self):
        print("\n" + "=" * 70)
        print("=== MODE POSITIONNEMENT RELATIF (µm depuis position actuelle) ===")
        print("→ Entre un déplacement en µm : +100 (avancer) ou -100 (reculer)")
        print("→ Exemples : +1000 = avancer de 1mm, -500 = reculer de 0.5mm")
        print("Commandes: nombre | 'p' (position) | '0' (retour home) | 'q' (quitter)")
        print("=" * 70)
        while True:
            self.master.send_processdata()
            self.master.receive_processdata(1000)
            inp = InputPdo.from_buffer_copy(self.device.input)
            cur_dev = inp.position_actual_value
            if self.COUNTS_PER_UM:
                cur_um_from_home = (cur_dev - self.HOME_POSITION_OFFSET) / (self.outward_sign * self.COUNTS_PER_UM)
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
                dev_target = self.HOME_POSITION_OFFSET
                self.pp_move_absolute(dev_target)
                continue
            try:
                if cmd.startswith('+'):
                    delta_um = float(cmd[1:])
                elif cmd.startswith('-'):
                    delta_um = -float(cmd[1:])
                else:
                    delta_um = float(cmd)
                self.move_relative_um(delta_um)
            except ValueError:
                print("❌ Entre un nombre (+100, -50, etc.), 'p', '0' ou 'q'.")

    def open(self):
        self.master.open(self.IFACE_NAME)
        if self.master.config_init() <= 0:
            print("❌ No device")
            self.master.close()
            return False
        self.device = self.master.slaves[0]
        self.clear_faults_via_sdo()
        time.sleep(0.2)
        self.device.config_func = self.config_func
        self.master.config_map()
        if self.master.state_check(pysoem.SAFEOP_STATE, 50_000) != pysoem.SAFEOP_STATE:
            print('❌ SAFEOP_STATE failed')
            self.master.close()
            return False
        self.master.state = pysoem.OP_STATE
        self.pd_thread = threading.Thread(target=self.processdata_thread, daemon=True)
        self.pd_thread.start()
        self.master.send_processdata()
        self.master.receive_processdata(2000)
        self.master.write_state()
        self.master.state_check(pysoem.OP_STATE, 5_000_000)
        if self.master.state != pysoem.OP_STATE:
            print('❌ OP_STATE failed')
            self.pd_thread_stop_event.set()
            self.pd_thread.join()
            self.master.close()
            return False
        return True

    def home(self):
        print("\n" + "="*70)
        print("=== VÉRIFICATION DES FAULTS ===")
        print("="*70)
        if self.check_faults():
            self.clear_faults_via_pdo()
            time.sleep(0.3)
            if self.check_faults():
                print("❌ Fault persistant, abandon.")
                return False
        outp = OutputPdo()
        outp.modes_of_operation = modes_of_operation['Homing mode']
        outp.controlword = 0x0006
        self.push_output(outp)
        time.sleep(0.2)
        outp.controlword = 0x0007
        self.push_output(outp)
        time.sleep(0.2)
        outp.controlword = 0x000F
        self.push_output(outp)
        time.sleep(0.3)
        print("\n" + "="*70)
        print("=== HOMING (driver) : instrumentation ===")
        print("="*70)
        outp.modes_of_operation = modes_of_operation['Homing mode']
        self.push_output(outp)
        outp.controlword = 0x003F
        self.push_output(outp)
        time.sleep(0.1)
        outp.controlword = 0x000F
        self.push_output(outp)
        t0 = time.time()
        homed_pos = None
        try:
            b = self.device.sdo_read(0x2604, 0, 4)
            steps_per_rev = int.from_bytes(b, "little", signed=False)
        except Exception:
            steps_per_rev = None
        if steps_per_rev and steps_per_rev > 0:
            counts_per_rev = steps_per_rev * self.SCALE_COUNTS
        else:
            counts_per_rev = 400 * self.SCALE_COUNTS
        if self.LEAD_MM_PER_REV <= 0:
            raise ValueError("LEAD_MM_PER_REV doit être > 0 (mm par tour).")
        self.COUNTS_PER_UM = counts_per_rev / (self.LEAD_MM_PER_REV * self.MICRONS_PER_MM)
        print(f"→ Mapping linéaire: COUNTS_PER_UM = {self.COUNTS_PER_UM:.6f} counts/µm  "
              f"(steps/rev={steps_per_rev or 400}, SCALE={self.SCALE_COUNTS}, lead={self.LEAD_MM_PER_REV} mm/rev)")
        while time.time() - t0 < 45:
            self.master.send_processdata()
            self.master.receive_processdata(1000)
            inp = InputPdo.from_buffer_copy(self.device.input)
            sw = inp.statusword
            pos = inp.position_actual_value
            if (sw & (1 << 12)) and (sw & (1 << 10)):
                print(f"✅ HOMING ATTAINED @ pos={pos}  (SW=0x{sw:04X})")
                homed_pos = pos
                self.HOME_POSITION_OFFSET = pos
                break
            time.sleep(0.02)
        if homed_pos is None:
            print("❌ Homing timeout")
            return False
        self.set_profile_position_params(vel=5000, acc=20000, dec=20000)
        outp_pp = OutputPdo()
        outp_pp.modes_of_operation = modes_of_operation['Profile position mode']
        self.cw_base_enable(outp_pp)
        outp_pp.target_position = homed_pos + self.PROBE_COUNTS
        self.push_output(outp_pp)
        time.sleep(0.005)
        self.push_output(outp_pp)
        self.wait_tr_fall(0.25)
        self.send_pp_front(outp_pp, relative=False)
        time.sleep(0.2)
        plus_ok = self.read_home_bit()
        if plus_ok == 1:
            detected_away_sign = +1
        else:
            outp_pp.target_position = homed_pos
            self.push_output(outp_pp)
            time.sleep(0.005)
            self.push_output(outp_pp)
            self.wait_tr_fall(0.25)
            self.send_pp_front(outp_pp, relative=False)
            time.sleep(0.2)
            outp_pp.target_position = homed_pos - self.PROBE_COUNTS
            self.push_output(outp_pp)
            time.sleep(0.005)
            self.push_output(outp_pp)
            self.wait_tr_fall(0.25)
            self.send_pp_front(outp_pp, relative=False)
            time.sleep(0.2)
            minus_ok = self.read_home_bit()
            detected_away_sign = -1 if minus_ok == 1 else +1
        self.outward_sign = detected_away_sign if self.LOCK_LOGIC_PLUS_IS_AWAY else detected_away_sign
        print(f"→ Sens imposé: '+' (logique) = s'éloigner ; outward_sign = {self.outward_sign:+d}")
        outp_pp.target_position = homed_pos + self.outward_sign * self.RETRAIT_COUNTS
        self.push_output(outp_pp)
        time.sleep(0.005)
        self.push_output(outp_pp)
        self.wait_tr_fall(0.25)
        self.send_pp_front(outp_pp, relative=False)
        time.sleep(0.25)
        self.master.send_processdata()
        self.master.receive_processdata(1000)
        cur_after = InputPdo.from_buffer_copy(self.device.input).position_actual_value
        self.phase_offset = cur_after % self.SCALE_COUNTS
        print(f"→ phase_offset = {self.phase_offset} (pos % {self.SCALE_COUNTS})")
        print(f"État capteur après dégagement: {self.read_home_bit()} (1=relâché)")
        return True

    def run_interactive(self):
        try:
            self.interactive_mode_um()
        except KeyboardInterrupt:
            print("\n🛑 Interruption utilisateur")

    def close(self):
        print("\nDésactivation...")
        outp = OutputPdo()
        outp.controlword = 0x0006
        self.push_output(outp)
        time.sleep(0.1)
        self.device.output = bytes(len(self.device.output))
        self.master.send_processdata()
        self.master.receive_processdata(1000)
        self.pd_thread_stop_event.set()
        self.pd_thread.join()
        self.master.state = pysoem.PREOP_STATE
        self.master.write_state()
        self.master.close()
        print("✓ Arrêt propre")

# Exemple d'utilisation pour une séquence de mouvements
if __name__ == "__main__":
    controller = Motor_controller()
    if controller.open():
        if controller.home():
            # Exemple de séquence : liste de dictionnaires avec delta_um, et optionnellement vel, acc, dec
            sequence = [
                {'delta_um': 1000, 'vel': 200, 'acc': 6000, 'dec': 6000},  # Avancer de 1mm à 200 um/s
                {'delta_um': -500},  # Reculer de 0.5mm avec params par défaut
                {'delta_um': 2000, 'vel': 150}  # Avancer de 2mm à 150 um/s
            ]
            controller.perform_sequence(sequence)
            controller.run_interactive()
    controller.close()