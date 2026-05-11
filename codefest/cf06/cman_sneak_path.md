# CMAN

Cell resistances: R[0][0] = 1 kΩ, R[0][1] = 2 kΩ, R[1][0] = 2 kΩ, R[1][1] = 1 kΩ

---

## (a) Ideal I_col0

Conditions: V_row0 = 1 V, V_row1 = 0 V, col 0 = 0 V, col 1 = 0 V.

Row 1 is grounded so only R[0][0] contributes:

    I_col0 = 1 V / 1 kΩ = 1.0 mA

---

## (b) KCL

Conditions: V_row0 = 1 V, col 0 = 0 V, row 1 and col 1 floating.

**KCL at V_row1** :

    (V_col1 - V_row1) / 1 kΩ = V_row1 / 2 kΩ
    
    V_row1 = (2/3) V_col1  

**KCL at V_col1** :

    (1 - V_col1) / 2 kΩ = (V_col1 - V_row1) / 1 kΩ
    
     1 + 2·V_row1 = 3·V_col1

Substituting (1) into (2):

    1 + (4/3)·V_col1 = 3·V_col1
    V_col1 = 3/5 = 0.6 V
    V_row1 = (2/3)(0.6) = 0.4 V

---

## (c) Actual I_col0 with sneak path

Row 1 floated up to 0.4 V, so both rows now source current into col 0:

    I_direct = 1.0 V / 1 kΩ = 1.0 mA  
    I_sneak  = 0.4 V / 2 kΩ = 0.2 mA  

    I_col0 = 1.0 + 0.2 = 1.2 mA  (20% error vs ideal)

Sneak route: row 0 → R[0][1] → col 1 → R[1][1] → row 1 → R[1][0] → col 0.

---

## (d) Why sneak paths corrupt MVM

Col 0 is supposed to only see current from its own column weights, but floating nodes let current take unintended routes through neighboring cells and add extra current that doesn't correspond to any real weight. In a larger array there are far more of these routes, so the errors compound and the output becomes meaningless.
