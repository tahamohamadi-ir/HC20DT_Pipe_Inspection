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

## Inner pipe wall

The pipe has an inner radius of 175 mm. With the scanner centred at
y = 0 and parallel to the pipe axis, its upper corner is 50 mm to the
side and z + 50 mm above the pipe centre. For a circular bore, first
contact occurs at z = √(175² − 50²) − 50 ≈ 117.7 mm.

I set the scanner centre to x = 300 mm so its rear face was at x = 50 mm,
clear of the collar ending at x = 25 mm. In the 32-segment pipe model,
I observed no contact at z = 118 mm and contact at z = 119 mm. These two
poses bracket the contact threshold for this orientation.