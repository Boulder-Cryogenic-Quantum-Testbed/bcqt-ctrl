import socket

# broadcast address is the bitwise OR between IP and bit-complement of the subnet mask
addr_SND = ('192.168.0.255', 4950)  # broadcast address / MCL test equipment listens on port 4950
addr_RCV = ('', 4951)               # MCL test equipment replies on port 4951
 
UDPSock_SND = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Create socket
UDPSock_RCV = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Create socket

UDPSock_RCV.bind(addr_RCV)
UDPSock_RCV.settimeout(1)
Data_RCV=""

all_snds = [str.encode("MCLRFSWITCH?"), str.encode("MCLDAT?")]    # Query for the relevant product family (encoded to bytes)

num_devices = 3

for Data_SND in all_snds:
    print("Sending message '{Data_SND}'")
    UDPSock_SND.sendto(Data_SND, addr_SND)

    print(f"    Listening for up to {num_devices} devices...")
    
    i=0
    while i<num_devices:                          # Search for up to x units

        print(f"    Device {i+1}")
        
        try:
            Data_RCV, addr_RCV = UDPSock_RCV.recvfrom(4951)
            print(str(Data_RCV).replace('\\r\\n','\n        ').replace("b'Model Name", "        Model Name"))

        except:                         # Timeout error if no more responses
            print("    No data received.")
            
        i=i+1

print("End of UDP listening...")

print('Client stopped.')


UDPSock_SND.close()             # Close sockets
UDPSock_RCV.close()
