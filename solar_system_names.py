''' 
These python lists contain the names and masses of bodies in the solar system. 
All masses here are in kilograms. 

Written by Pablo Lemos (UCL)
pablo.lemos.18@ucl.ac.uk
19-11-2020
'''

sun_mass = 1.9885e+30

planet_names = ['mercury', 
                'venus',
                'earth', 
                'mars',
                'jupiter',
                'saturn', 
                'uranus', 
                'neptune', 
                ]

planet_masses = [0.33011 * 10**24, 
                4.8685 * 10**24,
                5.9724 * 10**24, 
                0.64171 * 10**24,
                1898.19 * 10**24,
                568.34 * 10**24, 
                86.81 * 10**24, 
                102.4 * 10**24, 
                ]

planets_with_moons = ['earth', 
#                'mars',
                'jupiter',
                'saturn', 
                #'uranus', 
                #'neptune', 
                ]


earth_moon_names = ['moon']

earth_moon_masses = [0.0734767309 * 10**24]

mars_moon_names = ['phobos', 
                   'deimos']

mars_moon_masses = [1.0659 * 10**16,
                   1.4762 * 10**15
]

jupiter_moon_names = ['io', 
                     'europa', 
                     'ganymede', 
                     'callisto', 
                    ]

jupiter_moon_masses = [0.08931900 * 10**24, 
                            0.048 * 10**24, 
                            0.14819 * 10**24,
                            0.10759 * 10**24]

saturn_moon_names = ['mimas', 
                     'enceladus', 
                     'tethys', 
                     'dione', 
                     'rhea', 
                     'titan', 
                     'hyperion', 
                     'iapetus',
                     'phoebe']

saturn_moon_masses = [0.000037493 * 10**24, 
                     0.000108022 * 10**24, 
                     0.000617449 * 10**24, 
                     0.001095452 * 10**24, 
                     0.002306518 * 10**24, 
                     0.1353452 * 10**24, 
                     5.6199 * 10**18, 
                     0.001805635 * 10**24,
                     8.292 * 10**18]

moon_names = [earth_moon_names, 
                   #mars_moon_names,
                   jupiter_moon_names,
                   saturn_moon_names]
        
moon_masses = [earth_moon_masses, 
                   #mars_moon_masses,
                   jupiter_moon_masses,
                   saturn_moon_masses]