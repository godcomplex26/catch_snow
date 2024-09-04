import serial.tools.list_ports

def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    port_list = []
    
    for port in ports:
        port_info = {
            "device": port.device,  # 포트 이름 (예: COM1, /dev/ttyUSB0)
            "name": port.name,      # 포트 이름 (대부분 동일함)
            "description": port.description,  # 포트에 대한 설명 (예: USB Serial Port)
            "hwid": port.hwid       # 하드웨어 ID
        }
        port_list.append(port_info)
    
    return port_list

def select_serial_port():
    available_ports = list_serial_ports()
    
    if not available_ports:
        print("No serial ports found.")
        return None
    
    print("Available serial ports:")
    for i, port in enumerate(available_ports):
        print(f"{i+1}: {port['device']} - {port['description']}")
    
    print(f"{len(available_ports) + 1}: None (Do not select a port)")

    while True:
        try:
            selection = int(input("Select a port by number: "))
            if 1 <= selection <= len(available_ports):
                return available_ports[selection - 1]
            elif selection == len(available_ports) + 1:
                return None
            else:
                print(f"Please select a valid number between 1 and {len(available_ports) + 1}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
if __name__ == "__main__":
    selected_port = select_serial_port()
    
    if selected_port:
        print(f"You selected: {selected_port['device']}")
        print(f"   Name: {selected_port['name']}")
        print(f"   Description: {selected_port['description']}")
        print(f"   HWID: {selected_port['hwid']}")
    else:
        print("No port selected.")
