"""
Tests for the agent. Gemini is MOCKED, so these run offline, instantly, and cost no API quota.

Run with:   python manage.py test agent -v 2
"""
from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from accounts.models import CustomUser
from agent.models import AgentDecision
from logging_app.models import WeightEntry, DailyCheckIn
from ml_layer.agent_module import planner


def fake_response(*calls):
    """Build a fake Gemini response. Each call is (tool_name, args_dict)."""
    return SimpleNamespace(
        function_calls=[SimpleNamespace(name=n, args=a) for n, a in calls],
        usage_metadata=None,
        text="",
    )


class FakeClient:
    def __init__(self, response=None, error=None):
        self._response, self._error = response, error
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.last_kwargs = kwargs
        if self._error:
            raise self._error
        return self._response


def make_user(goal="lose", adherence=5, first_w=80.0, last_w=81.0, **extra):
    user = CustomUser.objects.create_user(username="t_" + goal + str(adherence), password="x" * 10, goal=goal,
                                          current_calorie_target=2000, **extra)
    for w in (first_w, last_w):
        WeightEntry.objects.create(user=user, weight_kg=w)
    DailyCheckIn.objects.create(user=user, adherence_score=adherence)
    return user


def run(user, response=None, error=None):
    client = FakeClient(response, error)
    with patch.object(planner, "_get_client", return_value=client):
        decision = planner.run_weekly_agent(user)
    return decision, client


class AgentToolSelectionTests(TestCase):
    """Different data -> Gemini picks different tools -> different outcomes."""

    def test_low_adherence_flags_and_keeps_calories(self):
        user = make_user(adherence=1)
        d, _ = run(user, fake_response(("flag_for_review", {"reason": "Adherence is very low."})))
        self.assertEqual(d.decision_type, "flag_for_review")
        self.assertEqual(d.new_calorie_target, 2000)
        self.assertEqual(d.tools_called[0]["tool"], "flag_for_review")

    def test_gaining_while_losing_decreases_calories(self):
        user = make_user(adherence=5)
        d, _ = run(user, fake_response(("adjust_macros", {
            "adjustment_type": "decrease", "percentage_change": 10.0, "current_calories": 2000,
            "reason": "Weight is rising while your goal is loss."})))
        self.assertEqual(d.decision_type, "decrease_calories")
        self.assertEqual(d.new_calorie_target, 1800)
        user.refresh_from_db()
        self.assertEqual(user.current_calorie_target, 1800)

    def test_on_track_maintains_and_suggests_recipe(self):
        user = make_user(goal="lose", adherence=5, first_w=80.0, last_w=79.0)
        d, _ = run(user, fake_response(
            ("adjust_macros", {"adjustment_type": "maintain", "percentage_change": 0.0, "reason": "On track."}),
            ("suggest_recipe", {"goal": "lose", "calorie_limit": 2000}),
        ))
        self.assertEqual(d.decision_type, "maintain_calories")
        self.assertEqual([t["tool"] for t in d.tools_called], ["adjust_macros", "suggest_recipe"])


class AgentSafetyFallbackTests(TestCase):
    def test_api_exception_falls_back_to_flag(self):
        d, _ = run(make_user(), error=RuntimeError("boom"))
        self.assertEqual(d.decision_type, "flag_for_review")
        self.assertEqual(d.new_calorie_target, d.previous_calorie_target)
        self.assertIn("error", [s["kind"] for s in d.trace])

    def test_no_tool_call_falls_back_to_flag(self):
        d, _ = run(make_user(), fake_response())
        self.assertEqual(d.decision_type, "flag_for_review")

    def test_unknown_tool_is_ignored_and_flagged(self):
        d, _ = run(make_user(), fake_response(("delete_everything", {})))
        self.assertEqual(d.decision_type, "flag_for_review")

    def test_bad_arguments_do_not_crash(self):
        d, _ = run(make_user(), fake_response(("adjust_macros", {"adjustment_type": "decrease",
                                                                  "percentage_change": "banana"})))
        self.assertEqual(d.decision_type, "flag_for_review")

    def test_percentage_is_clamped_by_tool(self):
        user = make_user()
        d, _ = run(user, fake_response(("adjust_macros", {"adjustment_type": "decrease",
                                                          "percentage_change": 90.0, "reason": "x"})))
        self.assertEqual(d.new_calorie_target, 1700)  # max 15% of 2000


class TimelineTests(TestCase):
    def test_trace_records_steps_in_order(self):
        user = make_user()
        d, _ = run(user, fake_response(("adjust_macros", {"adjustment_type": "decrease",
                                                          "percentage_change": 5.0, "reason": "Trending up."})))
        kinds = [s["kind"] for s in d.trace]
        self.assertEqual(kinds, ["observe", "offer", "model", "call", "result", "decision"])
        offer = next(s for s in d.trace if s["kind"] == "offer")
        called = {t["name"]: t["called"] for t in offer["data"]["tools"]}
        self.assertEqual(called, {"adjust_macros": True, "flag_for_review": False, "suggest_recipe": False})

    def test_detail_page_renders_timeline(self):
        user = make_user()
        d, _ = run(user, fake_response(("flag_for_review", {"reason": "Check in needed."})))
        self.client.force_login(user)
        resp = self.client.get(reverse("agent:decision_detail", args=[d.id]))
        self.assertContains(resp, "Agent reasoning timeline")
        self.assertContains(resp, "flag_for_review")
        self.assertContains(resp, "Tools offered to Gemini")

    def test_old_decision_without_trace_still_renders(self):
        user = make_user()
        d = AgentDecision.objects.create(user=user, decision_type="maintain_calories", previous_calorie_target=2000,
                                         new_calorie_target=2000, tools_called=[{"tool": "adjust_macros", "args": {"a": 1}}])
        self.client.force_login(user)
        self.assertEqual(self.client.get(reverse("agent:decision_detail", args=[d.id])).status_code, 200)

    def test_other_users_decision_is_404_not_500(self):
        owner, intruder = make_user(goal="lose"), CustomUser.objects.create_user(username="other", password="x" * 10)
        d, _ = run(owner, fake_response(("flag_for_review", {"reason": "r"})))
        self.client.force_login(intruder)
        self.assertEqual(self.client.get(reverse("agent:decision_detail", args=[d.id])).status_code, 404)
