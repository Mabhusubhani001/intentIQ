from features_model2 import extract_features_model2

# ============================================================
# GUARDRAIL THRESHOLDS
# ============================================================
# Hesitation: hovering in betslip > 2.5s without betting
CTA_DWELL_HESITATION_MS = 2500

# Erratic: clicking the bet button more than 1 time per second
BET_CLICK_VELOCITY_ERRATIC = 1.0


class IntentRiskMonitor:
    """
    Rule-based guardrail for Model 2.
    No ML model needed — logic is explicit and auditable.
    
    Signals:
      NONE               -> Normal behaviour, user is browsing/betting normally
      SHOW_CLARITY_CARD  -> Hesitation detected (hovering on bet button without clicking)
      SHOW_SAFETY        -> Erratic detected (rapid spamming of the bet button)
    """

    def __init__(self, model_path=None):
        # model_path is ignored — we use rule-based logic
        print("Initializing Model 2: Intent & Risk Monitor (rule-based guardrail)...")

    def predict(self, events):
        features = extract_features_model2(events)

        cta_dwell = features['cta_dwell_time']
        bet_clicks_per_sec = features['bet_click_velocity']

        # --- Decision logic (strictly ordered by priority) ---
        # 1. Erratic: rapid bet-button clicking
        if bet_clicks_per_sec >= BET_CLICK_VELOCITY_ERRATIC:
            signal = "SHOW_SAFETY"
            message = "Let's take a moment — you've been placing bets quickly. Take your time."

        # 2. Hesitation: long hover in betslip without committing
        elif cta_dwell >= CTA_DWELL_HESITATION_MS:
            signal = "SHOW_CLARITY_CARD"
            message = "This locks in your selected odds."

        # 3. Normal behaviour
        else:
            signal = "NONE"
            message = ""

        return {
            "signal": signal,
            "message": message,
            "model2_features": features
        }
