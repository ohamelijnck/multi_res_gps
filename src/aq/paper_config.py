def get_config():
    return  {
        'models': {
            'mr_dgp': {
                'line_color': 'red',
                'fill_color': 'red',
                'line_style': 'solid',
                'pretty_name': r'\cmgpdgp',
            },
            'mr_gprn': {
                'line_color': '#225ea8',
                'fill_color': '#225ea8',
                'line_style': 'dashed',
                'pretty_name': r'\cmgpaggr',
            },
            'mr_gprn_pos_w': {
                'line_color': 'orange',
                'fill_color': 'orange',
                'line_style': '-.',
                'pretty_name': r'\cmgpaggrpos',
            },
            'mean_baseline': {
                'line_color': '#c2e699',
                'fill_color': '#c2e699',
                'line_style': (0, [1, 1]),
                'pretty_name': 'Mean Baseline',
            },
            'center_baseline': {
                'line_color': '#969696',
                'fill_color': '#969696',
                'line_style': (0, [1, 3]),
                'pretty_name': 'Center Point Baseline',
            },
            'cascade_dgp_baseline': {
                'line_color': '#c2e699',
                'fill_color': '#c2e699',
                'line_style': (0, [1, 1]),
                'pretty_name': 'Cascade MR-DGP',
            },
            'naive_dgp_baseline': {
                'line_color': '#969696',
                'fill_color': '#969696',
                'line_style': (0, [1, 3]),
                'pretty_name': 'Naive MR-DGP',
            },
            'baseline': {
                'line_color': 'grey',
                'fill_color': 'red',
                'line_style': 'solid',
                'pretty_name': 'baseline',
            },

            'vbagg': {
                'line_color': 'grey',
                'fill_color': 'red',
                'line_style': 'solid',
                'pretty_name': r'\vbagg',
            },
        },
        'observations': {
            'point': {
                'color': 'grey',
                'style': 'x',
            },
            'line': {
                'color': 'grey',
                'style': 'solid',
            }

        },
        'observed_target': {
            'point': {
                'color': 'grey',
                'style': 'x',
            },
            'line': {
                'color': 'grey',
                'style': 'solid',
            }

        },
        'latex_preamble': r'\newcommand{\cmgpdgp}{MR-DGP} \newcommand{\cmgpgp}{MR-GP} \newcommand{\cmgpaggr}{MR-GPRN} \newcommand{\cmgpaggrpos}{mr-pos-gprn} \newcommand{\vbagg}{VBAGG-NORMAL}',
    }


