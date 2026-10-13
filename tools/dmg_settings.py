# dmgbuild settings for the macOS installer window.
# Usage: dmgbuild -s tools/dmg_settings.py -D app=dist/Autotune.app \
#          -D background=bg.tiff -D icon=volume.icns "Автотюн" Autotune.dmg
import os

app = defines["app"]  # noqa: F821 (provided by dmgbuild)
app_name = os.path.basename(app)

format = "UDZO"
files = [app]
symlinks = {"Applications": "/Applications"}
icon = defines.get("icon")  # noqa: F821
background = defines["background"]  # noqa: F821

# keep in sync with tools/make_dmg_background.py (APP_X, APPS_X, ICON_Y)
icon_locations = {app_name: (180, 190), "Applications": (480, 190)}
window_rect = ((200, 120), (660, 440))
default_view = "icon-view"
icon_size = 128
text_size = 13
show_status_bar = False
show_tab_view = False
show_toolbar = False
show_pathbar = False
show_sidebar = False
show_icon_preview = False
