import time

import viser

from app import PentosApp
from theming import add_build_plate_scene, configure_theme


server = viser.ViserServer(label="Pentos")
configure_theme(server)
add_build_plate_scene(server)

app = PentosApp(server)
app.show_setup()

print(f"Open your browser to http://localhost:{server.get_port()}")
print("Press Ctrl+C to exit")

while True:
    time.sleep(10.0)
