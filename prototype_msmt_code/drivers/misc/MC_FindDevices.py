import socket

# broadcast address is the bitwise OR between IP and bit-complement of the subnet mask
addr_SND = ('192.168.0.255', 4950)  # broadcast address / MCL test equipment listens on port 4950
addr_RCV = ('', 4951)               # MCL test equipment replies on port 4951
 
UDPSock_SND = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Create socket
UDPSock_RCV = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Create socket

UDPSock_RCV.bind(addr_RCV)
UDPSock_RCV.settimeout(1)
Data_RCV=""

Data_SND = str.encode("MCLRFSWITCH?")    # Query for the relevant product family (encoded to bytes)
Data_SND = str.encode("MCLDAT?")    # Query for the relevant product family (encoded to bytes)

print ("Sending message '%s'..." % Data_SND)
UDPSock_SND.sendto(Data_SND, addr_SND)

print ("Listening for up to 5 devices...")
i=0
while i<5:                          # Search for up to 5 units

    print ("Device '%s'..." % str(i + 1))
    
    try:
        Data_RCV,addr_RCV = UDPSock_RCV.recvfrom(4951)
        print (Data_RCV)

    except:                         # Timeout error if no more responses
        print ("No data received.")
        
    i=i+1

print ("End of UDP listening...")

UDPSock_SND.close()             # Close sockets
UDPSock_RCV.close()

print ('Client stopped.')
