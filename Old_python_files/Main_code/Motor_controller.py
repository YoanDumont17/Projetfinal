import pysoem  # Bibliothèque pour la communication EtherCAT
import ctypes  # Bibliothèque pour définir des structures de données binaires compatibles avec C
import time  # Bibliothèque pour gérer les pauses et temporisations
import threading  # Bibliothèque pour créer des threads parallèles







class InputPdo(ctypes.Structure):  # Définit la structure pour les PDO d'entrée (données reçues du dispositif)
    _pack_ = 1              # Force l'alignement byte-à-byte sans padding pour matcher le protocole
    _fields_ = [            # Liste des champs de la structure avec leurs types C
        ('Errorword', ctypes.c_uint16),             # Code d'erreur du moteur sous forme d'entier non signé 16 bits, c'est-à-dire un nombre entier positif
        ('statusword', ctypes.c_uint16),            # Mot de statut du moteur sous forme d'entier non signé 16 bits
        ('position_actual_value', ctypes.c_int32),              # Position actuelle du moteur sous forme d'entier signé 32 bits, c'est-à-dire un nombre entier pouvant être positif ou négatif
        ('velocity_actual_value', ctypes.c_int32),              # Vitesse actuelle du moteur sous forme d'entier signé 32 bits
        ('followerrorcodevalue', ctypes.c_uint32)           # Code d'erreur de suivi du moteur sous forme d'entier non signé 32 bits
    ]







class OutputPdo(ctypes.Structure):  # Définit la structure pour les PDO de sortie (données envoyées au dispositif)
    _pack_ = 1              # Force l'alignement byte-à-byte sans padding
    _fields_ = [            # Liste des champs de la structure
        ('controlword', ctypes.c_uint16),           # Mot de contrôle (16 bits non signé)
        ('modes_of_operation', ctypes.c_int8),              # Mode d'opération (8 bits signé)
        ('target_position', ctypes.c_int32),            # Position cible (32 bits signé)
        ('target_velocity', ctypes.c_int32),            # Vitesse cible (32 bits signé)
    ]




modes_of_operation = {              # Dictionnaire contenant  les différents modes d'opération à leurs codes numériques (standard CiA 402)
    'No mode': 0,                   # Pas de mode
    'Profile position mode': 1,     # Mode position profilée
    'Profile velocity mode': 3,     # Mode vitesse profilée
    'Profile_torque_mode': 4,       # Mode couple profilé
    'Homing mode': 6,               # Mode homing (recherche de zéro)
    'Cyclic synchronous position mode': 8,          # Mode position synchrone cyclique
    'Cyclic synchronous velocity mode': 9,          # Mode vitesse synchrone cyclique
    'Cyclic synchronous torque mode': 10,           # Mode couple synchrone cyclique
    'Q_mode': -1,                   # Mode spécifique au fabricant
}

class MotorController:  
    '''Cette classe gère la communication avec un contrôleur de moteur via EtherCAT. Elle permet d'ouvrir la connexion, de configurer le dispositif, de commander des mouvements, et de fermer proprement la connexion.'''



    def __init__(self, iface_name="enP8p1s0"): 

        self.iface_name = iface_name                            # Stocke le nom de l'interface réseau EtherCAT du Jetson Nano (enP8p1s0)
        self.pd_thread_stop_event = threading.Event()           # Événement pour arrêter le thread de traitement PDO. Cela permet de signaler au thread de s'arrêter proprement. Le thread vérifiera périodiquement cet événement pour savoir s'il doit continuer à fonctionner ou s'arrêter.
        self.master = pysoem.Master()                           # Crée l'objet master EtherCAT
        self.actual_wkc = 0                                     # Compteur de "working counter" pour vérifier les échanges. Voir explication ci-dessous

        #Le master (ici, self.master = pysoem.Master()) envoie des trames de données (paquets) aux slaves (les dispositifs connectés, comme ton moteur).
        #Chaque trame passe par tous les slaves en chaîne, et chaque slave qui traite correctement la trame incrémente (augmente de 1) un compteur appelé Working Counter (WKC) dans la trame.
        # À la fin, le master reçoit la trame de retour et lit le WKC final. Ce WKC indique combien de slaves ont bien traité la trame.

        self.device = None                                      # Référence au slave (dispositif) EtherCAT
        self.proc_thread = None                                 # Référence au thread de traitement




    def configure(self):
        '''Méthode pour configurer les paramètres du dispositif via SDO. Les SDO sont un mécanisme pour lire ou écrire des données de configuration dans un dispositif connecté (comme un moteur ou un capteur). 
        Contrairement aux échanges rapides et cycliques (comme les PDO pour les données en temps réel), 
        les SDO sont utilisés pour des opérations non cycliques, comme paramétrer un appareil.'''



        # profile velocity  # Configure la vitesse de profil
        self.device.sdo_write(0x6081, 0, bytes(ctypes.c_int32(100)))  # Écrit la vitesse à l'adresse correspondante (x impulsions/tour). Dans notre cas, 200 = 1 tour/s

        # profile acceleration  # Configure l'accélération de profil
        self.device.sdo_write(0x6083, 0, bytes(ctypes.c_int32(50000)))  # 50000 unités

        # profile deceleration  # Configure la décélération de profil
        self.device.sdo_write(0x6084, 0, bytes(ctypes.c_int32(50000)))  # Écrit la décélération (50000 unités)




    def processdata_thread(self): 

        """ Cette fonction permet d'échanger en temps continu des PDO entre le driver et le Jetson. Bref, Échanger périodiquement les données PDO (envoi et réception) pour que le système reste synchronisé.
          EtherCAT est un protocole en temps réel, donc ce thread tourne en boucle pour envoyer des commandes et recevoir des états (comme la position ou la vitesse du moteur). """

        while not self.pd_thread_stop_event.is_set():                   # Boucle tant que l'événement d'arrêt n'est pas activé
            self.master.send_processdata()                              # Envoie les données PDO
            self.actual_wkc = self.master.receive_processdata(10000)    # Reçoit les données avec timeout de 10000 µs
            if not self.actual_wkc == self.master.expected_wkc:         # Vérifie si le working counter est correct
                print('incorrect wkc')                                  # Affiche une erreur si non
            time.sleep(0.01)                                            # Pause de 0.01 seconde avant la prochaine itération

    def open(self):  

        """Méthpde qui effectue l'ouverture de la connection EtherCAT avec le driver"""
        self.master.open(self.iface_name)                               # Ouvre de l'interface réseau spécifiée (Jetson Nano)
        if self.master.config_init() > 0:                               # Initialise la configuration ; >0 signifie au moins un slave trouvé
            self.device = self.master.slaves[0]                         # Récupère le premier slave (dispositif)
            self.device.config_func = self.configure                    # Assigne la fonction de configuration au slave. Voir la méthode configue plus haut
            self.master.config_map()                                    # Mappe les PDO pour l'échange de données

            if self.master.state_check(pysoem.SAFEOP_STATE, 50_000) == pysoem.SAFEOP_STATE:  # Vérifie le passage en SAFEOP (timeout 50 ms)
                self.master.state = pysoem.OP_STATE  # Demande le passage en OP (opérationnel)

                self.proc_thread = threading.Thread(target=self.processdata_thread)  # Crée le thread de traitement PDO
                self.proc_thread.start()  # Lance le thread

                self.master.send_processdata()  # Envoie les données PDO (initial)
                self.master.receive_processdata(2000)  # Reçoit avec timeout de 2 ms

                self.master.write_state()  # Écrit l'état demandé
                self.master.state_check(pysoem.OP_STATE, 5_000_000)  # Vérifie le passage en OP (timeout 5 s)

                if self.master.state == pysoem.OP_STATE:  # Si succès
                    return True  # Retourne True
                else:  # Sinon
                    print('failed to go to OP_STATE')  # Affiche l'erreur
                    return False  # Retourne False
            else:  # Si échec SAFEOP
                print('failed to go to safeop state')  # Affiche l'erreur
                return False  # Retourne False
        else:  # Si aucun slave trouvé
            print('no device found')  # Affiche l'erreur
            return False  # Retourne False

    def move_to_position(self, position=3050):  # Méthode pour commander un mouvement à une position absolue (défaut 3050 counts)
        output_data = OutputPdo()  # Crée un objet pour les données de sortie
        output_data.modes_of_operation = modes_of_operation['Profile position mode']  # Définit le mode PP
        output_data.target_position = position  # Définit la position cible (ex. : 20000 = 1 tour/sec selon scaling)

        for control_cmd in [6, 7, 15, 31]:  # Séquence de commandes : shutdown, switch on, enable, new set-point (31 pour valider position en PP)
            output_data.controlword = control_cmd  # Met à jour le mot de contrôle
            self.device.output = bytes(output_data)  # Convertit en bytes et assigne aux données de sortie du slave
            self.master.send_processdata()  # Envoie les PDO
            self.master.receive_processdata(1_000)  # Reçoit avec timeout de 1 ms
            time.sleep(0.05)  # Pause de 0.05 s (augmenter si nécessaire pour la stabilité)

    def run(self):  # Méthode pour surveiller le mouvement en boucle
        try:  # Essaie
            while True:  # Boucle infinie
                self.master.send_processdata()  # Envoie PDO
                self.master.receive_processdata(1_000)  # Reçoit PDO
                time.sleep(0.05)  # Pause de 0.05 s
        except KeyboardInterrupt:  # Capture l'interruption clavier (Ctrl+C)
            print('stopped')  # Affiche "stopped"

    def close(self):  # Méthode pour fermer proprement la connexion
        # zero everything  # Réinitialise tout
        self.device.output = bytes(len(self.device.output))  # Met les données de sortie à zéro (bytes vides de même longueur)
        self.master.send_processdata()  # Envoie PDO
        self.master.receive_processdata(1_000)  # Reçoit PDO
        self.pd_thread_stop_event.set()  # Active l'événement d'arrêt du thread
        self.proc_thread.join()  # Attend la fin du thread

        self.master.state = pysoem.PREOP_STATE  # Demande le passage en PREOP
        self.master.write_state()  # Écrit l'état
        self.master.close()  # Ferme le master EtherCAT

if __name__ == "__main__":  # Bloc exécuté si le script est lancé directement
    controller = MotorController("enP8p1s0")  # Crée l'objet controller avec interface (remplacer par votre ID d'adaptateur)
    if controller.open():  # Si ouverture réussie
        controller.move_to_position(3050)  # Commande un mouvement à 3050 counts
        controller.run()  # Lance la surveillance en boucle
    controller.close()  # Ferme la connexion