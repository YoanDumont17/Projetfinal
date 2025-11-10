#Code communivation EtherCat

import pysoem
import time
import struct
import logging



master = pysoem.Master()   # Ici on crée une instance de Master pour gérer la communication EtherCAT
master.open()('eth0')  # Ouvre la communication sur l'interface réseau spécifiée (ici 'eth0')
master.slaves  # Donne la liste des esclaves trouvées durant l'initialisation. Le type est CdefSlave

master.sdo_read_timeout() # Delai d'attente pour la lecture SDO

master.sdo_write_timeout() # Delai d'attente pour l'écriture SDO

master.always_release_git() # True to always release the GIT on close

master.check_release_git() # Check if the GIT should be released : Parameters : release : bool

master.close() # Ferme la communication avec les esclaves

master.config_dc() # Locate DC slaves, measure propagation delays. Returns number of DC slaves found (bool)


master.config_init() #config_init(usetable=False, *, release_gil=None) Initialise la configuration des esclaves. Returns number of slaves found (int)
                     # usetable : bool : If True, use the saved configuration table if it exists.
                     # release_gil : bool : If True, release the GIL during the operation.
                     # Returns : int : Number of slaves found. -1 when no slaves found
            
master.config_map() #Map all slaves PDOs in IO map (PDO = Process Data Object). Returns number of mapped bytes (int)

master.config_overlap_map() #Map alls slaves PDOs to overlapping IO map. Returns number of mapped bytes (int)

master.dc_time  # DC times in nanoseconds to shynchronize the EtherCAT cycle with SYNCO cycles

master.expected_wkc() # Calculates the expected Working Counter ??

master.open() #Initialize and open network interface. On Linix On Linux the name of the interface is the same as usd by the system, e.g. eth0, and as displayed by ip addr

master.recerve_processdata() # Receive processdata from slaves. Second part from send_processdata(). Received datagrams are recombined with the processdata ...

master.state_check() # Check the state of all slaves. This is a blocking function. To refresh the state of all slaves read_state() should be called

master.write_state() # write all slaves state. The function does not check if the actual state is changed







cdefslave = pysoem.CdefSlave()  # Ici on crée une instance de CdefSlave pour gérer un esclave EtherCAT

cdefslave.add_emergency_callback() # get notified on EMCY messages from this slave

cdefslave.amend_mbx(mailbox, start_address, size)#change the start adresse size of a mailbox. Note that the slave must me in INIT state to do that.

cdefslave.config_func() # slaves callback function that is called during config_map

cdefslave.dc_sync() #Activate or deactivate SYNC pulses at the slave

cdefslave.eeprom_read(word_address, timeout = 20000) #Read 4 byte from EEPROM of the slave. Returns 4 byte as bytearray

cdefslave.eeprom_write(word_address, data, timeout = 20000) #Write 2 byte to EEPROM of the slave. Data must be a bytearray of length 2

cdefslave.foe_read(filename, password, size, timeout = 20000) #read given filename form device using File over EtherCAT (FoE). Returns data as bytearray FoE: Filename (string)-name of the target file
#passeword (int)-password for the file transfer
#size (int)-maximum file size

cdefslave.foe_write(filename, password, data, timeout = 20000) #write given filename to device using File over EtherCAT (FoE). Data must be a bytearray FoE: Filename (string)-name of the target file
#passeword (int)-password for the file transfer


cdefslave.get_sdo() # Get SDO object for given index and subindex. Returns SDO object

cdefslave.sdo_read(index, subindex, size=0, ca=False)

"""
Read a CoE object.

When leaving out the size parameter, objects up to 256 bytes can be read. If the size of the object is expected to be bigger, increase the size parameter.

Parameters
:
index (int) – Index of the object.

subindex (int) – Subindex of the object.

size (int, optional) – The size of the reading buffer.

ca (bool, optional) – complete access.

release_gil (bool, optional) – True to read a CoE object releasing the GIL. Defaults to False.

Returns
:
The content of the sdo object.

Return type
:
bytes

Raises
:
SdoError – if write fails, the exception includes the SDO abort code

MailboxError – on errors in the mailbox protocol

PacketError – on packet level error

WkcError – if working counter is not higher than 0, the exception includes the working counter
"""


cdefslave.sdo_write(index, subindex, data, ca=False)
'''
Write to a CoE object.

Parameters
:
index (int) – Index of the object.

subindex (int) – Subindex of the object.

data (bytes) – data to be written to the object.

ca (bool, optional) – complete access.

release_gil (bool, optional) – True to write to a CoE object releasing the GIL. Defaults to False.

Raises
:
SdoError – if write fails, the exception includes the SDO abort code

MailboxError – on errors in the mailbox protocol

PacketError – on packet level error

WkcError – if working counter is not higher than 0, the exception includes the working counter
'''


cdefslave.state  # Get or set the state of the slave. Possible states are defined in pysoem.ec_state
cdefslave.name  # Get the name of the slave as read from the EEPROM
cdefslave.state_check() # Check the state of the slave. This is a blocking function. To refresh the state of the slave read_state() should be called


helpers = pysoem.find_adapters()  # Cette fonction recherche les adaptateurs réseau disponibles pour la communication EtherCAT et renvoie une liste de ceux-ci.
pysoem.al_status_code_to_string()