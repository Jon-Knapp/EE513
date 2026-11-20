# Single-layer KiCad practice schematics (levels 1-8)

Eight schematics of increasing complexity, all through-hole basic parts,
designed so every board can be routed on ONE copper layer with no vias,
no inner layers, no jumpers and no trace running under/between a part's pads.

| # | Folder | Circuit | Parts | Nets |
|---|--------|---------|------:|-----:|
| 1 | 01_led_resistor | LED + current-limit resistor | 3 | 3 |
| 2 | 02_divider_rc_filter | Resistive divider + RC low-pass | 6 | 4 |
| 3 | 03_transistor_switch | Push-button NPN switch driving an LED | 7 | 6 |
| 4 | 04_common_emitter_amplifier | Divider-biased common-emitter amplifier | 12 | 7 |
| 5 | 05_astable_multivibrator | Two-transistor astable LED flasher | 12 | 8 |
| 6 | 06_opamp_dc_amplifier | LM358 single-supply DC amplifier, gain 11 | 13 | 8 |
| 7 | 07_bridge_rectifier_5v_supply | Diode bridge + 7805 regulated 5 V supply | 13 | 6 |
| 8 | 08_dark_activated_led_driver | LDR + NPN + MOSFET dark-activated LED driver | 19 | 11 |

Each folder has the `.kicad_sch`, a minimal `.kicad_pro`, and a PDF rendered by KiCad.
Files are in native KiCad 10 format (schematic format 20260306, generator_version "10.0"), the format every
KiCad 10.0.x release reads and writes. They will not open in KiCad 9 or older.
Symbols are embedded in each schematic; footprints point at the standard KiCad THT libraries.
PDFs were rendered by KiCad from the same drawings.

## How "single-layer routable" was checked
For each netlist, every part was modelled as a solid body with its pads in footprint
order around its outline (inline TO-92/TO-220 middle pads reachable from both sides,
DIP pads only from outside the package), every net as one copper node, and the
resulting graph tested for planarity. All eight pass. Connectivity of every file was
also confirmed with KiCad's own netlist export (kicad-cli 7.0.11).

## Design notes forced by the single-layer rule
- **05 astable:** powered from two separate wire pads (BAT+/BAT-). With a 2-pin
  connector forcing +5V and GND side by side, the cross-coupled netlist is non-planar.
- **06 op-amp:** only one LM358 section does real work (the other is a grounded
  follower). With V+ on pin 8 and V- on pin 4, nothing can pass between the pin 1-3
  and pin 5-7 sides of the DIP. The input is ground-referenced for the same reason:
  a Vcc/2 bias network on the input would trap pins 6-7 away from GND.
- **07 supply:** the AC connector J1 must sit inside the ring of four diodes (the
  bridge plus connector is a K4 graph), so both AC traces reach the bridge without crossing.
- **NE555 left out:** the astable needs pin 2-6 and pin 4-8 links, which sit on
  opposite sides of the DIP and always cross unless a trace runs under the chip.
