"""
==========================================
Fuzzy Control Systems: Speed ticket predictions
==========================================

Adam Kałuża s20831
Krzysztof Lewandowski s20491

To run program install
pip install scikit-fuzzy
pip install matplotlib

The program is used to predict fines for speeding tickets.

* Antecednets (Inputs)
   - `speed`
      * Vehicle speed between 50 and 140 km/h.
      * Divided into ranges that allow you to determine the amount of the mandate
      * Fuzzy set: 0-50, 50-90, 90-100, 100-120, 120-140, 140+
   - `permitted_speed`
      * Ranges of permissible speeds on Polish roads for passenger vehicles and trucks
      * Fuzzy set: 50, 90 ,120 ,140 and for trucks 50 ,70 ,80
    - `vehicle_weight`
      * Vehicle weight to determine if it is a passenger vehicle or truck
      * Fuzzy set: personal, truck
* Consequents (Outputs)
   - `ticket`
      * Universe: How much can we expect the mandate 0 - 2500zł
      * Fuzzy set: 0, 1, 2, 3, 4
* Rules
   - Applicable rules for maximum vehicle speeds
   - IF the *speed* is higher than *permitted_speed* we get a speed ticket prediction


Creating the Ticket predictions Using the skfuzzy control API
-------------------------------------------------------------
"""
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# New Antecedent/Consequent objects hold universe variables and membership
# functions
speed = ctrl.Antecedent(np.arange(0, 251, 1), 'speed')
permitted_speed = ctrl.Antecedent(np.arange(20, 141, 1), 'permitted_speed')
vehicle_weight = ctrl.Antecedent(np.arange(0, 28001, 1), 'vehicle_weight')
ticket = ctrl.Consequent(np.arange(0, 2501, 1), 'ticket')

# Custom membership functions

vehicle_weight['personal'] = fuzz.trimf(vehicle_weight.universe, [0, 1750, 3500])
vehicle_weight['truck'] = fuzz.trimf(vehicle_weight.universe, [3500, 15750, 28000])

speed['0-50'] = fuzz.trimf(speed.universe, [0, 25, 50])
speed['50-90'] = fuzz.trimf(speed.universe, [50, 70, 90])
speed['90-100'] = fuzz.trimf(speed.universe, [90, 95, 100])
speed['100-120'] = fuzz.trimf(speed.universe, [100, 110, 120])
speed['120-140'] = fuzz.trimf(speed.universe, [120, 130, 140])
speed['140+'] = fuzz.trimf(speed.universe, [140, 195, 250])

speed['50-70'] = fuzz.trimf(speed.universe, [50, 60, 70])
speed['70-80'] = fuzz.trimf(speed.universe, [70, 75, 80])
speed['80+'] = fuzz.trimf(speed.universe, [80, 145, 250])

permitted_speed['50'] = fuzz.trimf(permitted_speed.universe, [0, 25, 50])
permitted_speed['90'] = fuzz.trimf(permitted_speed.universe, [50, 70, 90])
permitted_speed['100'] = fuzz.trimf(permitted_speed.universe, [90, 95, 100])
permitted_speed['120'] = fuzz.trimf(permitted_speed.universe, [100, 110, 120])
permitted_speed['140'] = fuzz.trimf(permitted_speed.universe, [120, 130, 140])

permitted_speed['70'] = fuzz.trimf(permitted_speed.universe, [50, 60, 70])
permitted_speed['80'] = fuzz.trimf(permitted_speed.universe, [70, 75, 80])

ticket['0'] = fuzz.trimf(ticket.universe, [0, 0, 0])
ticket['1'] = fuzz.trimf(ticket.universe, [0, 250, 500])
ticket['2'] = fuzz.trimf(ticket.universe, [500, 750, 1000])
ticket['3'] = fuzz.trimf(ticket.universe, [1000, 1250, 1500])
ticket['4'] = fuzz.trimf(ticket.universe, [1500, 2000, 2500])

"""
To help understand what the membership looks like, use the ``view`` methods.
"""
speed.view()
permitted_speed.view()
vehicle_weight.view()
ticket.view()

"""
Fuzzy rules
-----------
"""
rule1 = ctrl.Rule((permitted_speed['50'] & speed['0-50']) & vehicle_weight['personal'], ticket['0'])
rule2 = ctrl.Rule((permitted_speed['50'] & speed['50-90']) & vehicle_weight['personal'], ticket['1'])
rule3 = ctrl.Rule((permitted_speed['50'] & speed['90-100']) & vehicle_weight['personal'], ticket['2'])
rule4 = ctrl.Rule((permitted_speed['50'] & speed['100-120']) & vehicle_weight['personal'], ticket['3'])
rule5 = ctrl.Rule((permitted_speed['50'] & speed['120-140']) & vehicle_weight['personal'], ticket['4'])
rule6 = ctrl.Rule((permitted_speed['50'] & speed['140+']) & vehicle_weight['personal'], ticket['4'])

rule7 = ctrl.Rule((permitted_speed['90'] & speed['0-50']) & vehicle_weight['personal'], ticket['0'])
rule8 = ctrl.Rule((permitted_speed['90'] & speed['50-90']) & vehicle_weight['personal'], ticket['0'])
rule9 = ctrl.Rule((permitted_speed['90'] & speed['90-100']) & vehicle_weight['personal'], ticket['1'])
rule10 = ctrl.Rule((permitted_speed['90'] & speed['100-120']) & vehicle_weight['personal'], ticket['2'])
rule11 = ctrl.Rule((permitted_speed['90'] & speed['120-140']) & vehicle_weight['personal'], ticket['3'])
rule12 = ctrl.Rule((permitted_speed['90'] & speed['140+']) & vehicle_weight['personal'], ticket['4'])

rule13 = ctrl.Rule((permitted_speed['100'] & speed['0-50']) & vehicle_weight['personal'], ticket['0'])
rule14 = ctrl.Rule((permitted_speed['100'] & speed['50-90']) & vehicle_weight['personal'], ticket['0'])
rule15 = ctrl.Rule((permitted_speed['100'] & speed['90-100']) & vehicle_weight['personal'], ticket['0'])
rule16 = ctrl.Rule((permitted_speed['100'] & speed['100-120']) & vehicle_weight['personal'], ticket['1'])
rule17 = ctrl.Rule((permitted_speed['100'] & speed['120-140']) & vehicle_weight['personal'], ticket['2'])
rule18 = ctrl.Rule((permitted_speed['100'] & speed['140+']) & vehicle_weight['personal'], ticket['3'])

rule19 = ctrl.Rule((permitted_speed['120'] & speed['0-50']) & vehicle_weight['personal'], ticket['0'])
rule20 = ctrl.Rule((permitted_speed['120'] & speed['50-90']) & vehicle_weight['personal'], ticket['0'])
rule21 = ctrl.Rule((permitted_speed['120'] & speed['90-100']) & vehicle_weight['personal'], ticket['0'])
rule22 = ctrl.Rule((permitted_speed['120'] & speed['100-120']) & vehicle_weight['personal'], ticket['0'])
rule23 = ctrl.Rule((permitted_speed['120'] & speed['120-140']) & vehicle_weight['personal'], ticket['1'])
rule24 = ctrl.Rule((permitted_speed['120'] & speed['140+']) & vehicle_weight['personal'], ticket['2'])

rule25 = ctrl.Rule((permitted_speed['140'] & speed['0-50']) & vehicle_weight['personal'], ticket['0'])
rule26 = ctrl.Rule((permitted_speed['140'] & speed['50-90']) & vehicle_weight['personal'], ticket['0'])
rule27 = ctrl.Rule((permitted_speed['140'] & speed['90-100']) & vehicle_weight['personal'], ticket['0'])
rule28 = ctrl.Rule((permitted_speed['140'] & speed['100-120']) & vehicle_weight['personal'], ticket['0'])
rule29 = ctrl.Rule((permitted_speed['140'] & speed['120-140']) & vehicle_weight['personal'], ticket['0'])
rule30 = ctrl.Rule((permitted_speed['140'] & speed['140+']) & vehicle_weight['personal'], ticket['2'])

rule34 = ctrl.Rule((permitted_speed['50'] & speed['0-50']) & vehicle_weight['truck'], ticket['0'])
rule35 = ctrl.Rule((permitted_speed['50'] & speed['50-70']) & vehicle_weight['truck'], ticket['1'])
rule36 = ctrl.Rule((permitted_speed['50'] & speed['70-80']) & vehicle_weight['truck'], ticket['2'])
rule37 = ctrl.Rule((permitted_speed['50'] & speed['80+']) & vehicle_weight['truck'], ticket['4'])

rule38 = ctrl.Rule((permitted_speed['70'] & speed['0-50']) & vehicle_weight['truck'], ticket['0'])
rule39 = ctrl.Rule((permitted_speed['70'] & speed['50-70']) & vehicle_weight['truck'], ticket['0'])
rule40 = ctrl.Rule((permitted_speed['70'] & speed['70-80']) & vehicle_weight['truck'], ticket['1'])
rule41 = ctrl.Rule((permitted_speed['70'] & speed['80+']) & vehicle_weight['truck'], ticket['2'])

rule42 = ctrl.Rule((permitted_speed['80'] & speed['0-50']) & vehicle_weight['truck'], ticket['0'])
rule43 = ctrl.Rule((permitted_speed['80'] & speed['50-70']) & vehicle_weight['truck'], ticket['0'])
rule44 = ctrl.Rule((permitted_speed['80'] & speed['70-80']) & vehicle_weight['truck'], ticket['0'])
rule45 = ctrl.Rule((permitted_speed['80'] & speed['80+']) & vehicle_weight['truck'], ticket['2'])

"""
Control System Creation and Simulation
---------------------------------------
"""

tipping_ctrl = ctrl.ControlSystem([
    rule1,
    rule2,
    rule3,
    rule4,
    rule5,
    rule6,
    rule7,
    rule8,
    rule9,
    rule10,
    rule11,
    rule12,
    rule13,
    rule14,
    rule15,
    rule16,
    rule17,
    rule18,
    rule19,
    rule20,
    rule21,
    rule22,
    rule23,
    rule24,
    rule25,
    rule26,
    rule27,
    rule28,
    rule29,
    rule30,
    rule34,
    rule35,
    rule36,
    rule37,
    rule38,
    rule39,
    rule40,
    rule41,
    rule42,
    rule43,
    rule44,
    rule45
])

ticket_predict = ctrl.ControlSystemSimulation(tipping_ctrl)

"""
We can now simulate our control system by simply specifying the inputs
"""
# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
ticket_predict.input['speed'] = 40
ticket_predict.input['permitted_speed'] = 51
ticket_predict.input['vehicle_weight'] = 2700

# Crunch the numbers
ticket_predict.compute()

"""
Once computed, we can view the result as well as visualize it.
"""
print(ticket_predict.output['ticket'])
ticket.view(sim=ticket_predict)

plt.show()
