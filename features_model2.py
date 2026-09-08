import numpy as np

def extract_features_model2(events):
    """
    Extract behavioral features from a rolling window of events.
    
    Specifically tracks:
    - cta_dwell_time: ms spent hovering in betslip/odds WITHOUT clicking the bet button
    - bet_click_velocity: clicks per second ON the betslip zone (bet button spam)
    - slip_jitter: micro-movement variance while inside betslip (hovering jitter)
    - zone_transitions_rolling: how often user hops between zones
    """
    if not events:
        return {
            'cta_dwell_time': 0.0,
            'bet_click_velocity': 0.0,
            'slip_jitter': 0.0,
            'zone_transitions_rolling': 0.0
        }

    t_min = events[0].get('t', 0)
    t_max = events[-1].get('t', 0)
    duration_s = max((t_max - t_min) / 1000.0, 1.0)

    # bet_click_velocity: clicks specifically on the betslip zone
    bet_clicks = sum(1 for e in events if e.get('e') == 'click' and e.get('z') in ('betslip', 'odds'))
    bet_click_velocity = bet_clicks / duration_s

    last_z = None
    zone_transitions = 0
    slip_moves = []
    dwell_time_ms = 0.0
    current_dwell_start = None

    for e in events:
        t = e.get('t', 0)
        z = e.get('z', 'background')

        # Zone transitions
        if z != last_z and last_z is not None:
            zone_transitions += 1
        last_z = z

        # Slip jitter: only mousemoves inside betslip
        if z == 'betslip' and e.get('e') == 'mousemove' and 'x' in e and 'y' in e:
            slip_moves.append((e['x'], e['y']))

        # CTA Dwell: time in betslip/odds without a bet click
        if z in ('betslip', 'odds'):
            if current_dwell_start is None:
                current_dwell_start = t
        else:
            if current_dwell_start is not None:
                dwell_time_ms += (t - current_dwell_start)
                current_dwell_start = None

        # A bet click resets the dwell (they committed, no longer hesitating)
        if e.get('e') == 'click' and z in ('betslip', 'odds'):
            if current_dwell_start is not None:
                dwell_time_ms += (t - current_dwell_start)
                current_dwell_start = None

    # Remaining dwell at end of window
    if current_dwell_start is not None:
        dwell_time_ms += (t_max - current_dwell_start)

    # Slip jitter: std deviation of step distances in betslip
    distances = []
    for i in range(1, len(slip_moves)):
        dx = slip_moves[i][0] - slip_moves[i-1][0]
        dy = slip_moves[i][1] - slip_moves[i-1][1]
        distances.append(np.sqrt(dx**2 + dy**2))

    slip_jitter = float(np.std(distances)) if len(distances) > 1 else 0.0

    return {
        'cta_dwell_time': float(dwell_time_ms),
        'bet_click_velocity': float(bet_click_velocity),
        'slip_jitter': float(slip_jitter),
        'zone_transitions_rolling': float(zone_transitions)
    }
