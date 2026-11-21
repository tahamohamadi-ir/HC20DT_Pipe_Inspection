# Collar contact check

The assignment specifies a 250 mm collar opening and a scanner with a
100 × 100 mm cross-section (assignment PDF, p. 2). I modelled the collar
with 32 boxes spaced 11.25° apart and kept the scanner parallel to the pipe axis.

For a circular opening, the upper corner reaches the rim when
50² + (z + 50)² = 125², with distances in millimetres. This gives a
scanner-centre height of 64.56 mm. I tested 60 mm and 70 mm on either
side of that value.

| Pose | Centre X (mm) | Centre Z (mm) | Expected contact | Observed contact points |
|---|---:|---:|---|---:|
| Fully outside | -260 | 0 | No | 0 |
| Centred in opening | -200 | 0 | No | 0 |
| Raised into rim | -200 | 85 | Yes | 20 |
| Upper corner clear | -200 | 60 | No | 0 |
| Upper corner intersects rim | -200 | 70 | Yes | 8 |

Run from the repository root: `python experiments/collar_contact.py`

Each check sets a static pose and calls `mj_forward`. A case passes when
the observed contact state (`data.ncon > 0`) matches the expected value
in the table. The point counts were recorded with MuJoCo 4.0.1.