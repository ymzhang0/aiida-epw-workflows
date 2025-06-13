from aiida.engine.processes.ports import PortNamespace
from aiida.engine.processes.builder import ProcessBuilderNamespace

def get_recursive_input_ports(namespace, print_value=False, indent=0):
    prefix = '  ' * indent
    for key, port in namespace.items():
        print(f"{prefix}- {key}")
        if isinstance(port, PortNamespace) or isinstance(port, ProcessBuilderNamespace):
            get_recursive_input_ports(port, print_value, indent + 1)
        elif print_value:
            print(f"{prefix}    {port}")