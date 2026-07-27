"""Tests for the profile-scoped credential primitive (Workstream A / Phase 2)."""
import pytest

from agent import secret_scope as ss


@pytest.fixture(autouse=True)
def _reset_multiplex():
    """Ensure each test starts and ends with multiplexing off (it's a global)."""
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


class TestMultiplexInactiveBackwardCompat:
    """Default deployment: get_secret transparently reads os.environ."""

    def test_reads_environ(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert ss.get_secret("ANTHROPIC_API_KEY") == "sk-test"

    def test_missing_returns_default(self, monkeypatch):
        monkeypatch.delenv("NOPE_KEY", raising=False)
        assert ss.get_secret("NOPE_KEY") is None
        assert ss.get_secret("NOPE_KEY", "fallback") == "fallback"

    def test_no_raise_without_scope(self, monkeypatch):
        monkeypatch.delenv("SOME_KEY", raising=False)
        # multiplex off => unscoped read is fine, returns default
        assert ss.get_secret("SOME_KEY") is None


class TestMultiplexActiveFailClosed:
    """Multiplex on: an unscoped secret read raises instead of leaking."""

    def test_unscoped_read_raises(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-leaky")
        ss.set_multiplex_active(True)
        with pytest.raises(ss.UnscopedSecretError):
            ss.get_secret("ANTHROPIC_API_KEY")

    def test_scoped_read_uses_scope_not_environ(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-environ")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-from-scope"})
        try:
            assert ss.get_secret("ANTHROPIC_API_KEY") == "sk-from-scope"
        finally:
            ss.reset_secret_scope(token)

    def test_scoped_missing_key_returns_default_not_environ(self, monkeypatch):
        # Even though the value exists in os.environ, a scope is authoritative:
        # an absent scope key must NOT fall through to the (cross-profile) env.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-other-profile")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-mine"})
        try:
            assert ss.get_secret("OPENAI_API_KEY") is None
            assert ss.get_secret("OPENAI_API_KEY", "d") == "d"
        finally:
            ss.reset_secret_scope(token)

    def test_global_env_still_reads_environ_under_multiplex(self, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", "/opt/data")
        ss.set_multiplex_active(True)
        # No scope, multiplex on — but HERMES_HOME is global, so no raise.
        assert ss.get_secret("HERMES_HOME") == "/opt/data"

    def test_kanban_prefix_is_global(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_DB", "/x/kanban.db")
        ss.set_multiplex_active(True)
        assert ss.get_secret("HERMES_KANBAN_DB") == "/x/kanban.db"


class TestScopeIsolation:
    """Two scopes never see each other's secrets."""

    def test_nested_scopes_restore(self):
        ss.set_multiplex_active(True)
        t1 = ss.set_secret_scope({"K": "a"})
        try:
            assert ss.get_secret("K") == "a"
            t2 = ss.set_secret_scope({"K": "b"})
            try:
                assert ss.get_secret("K") == "b"
            finally:
                ss.reset_secret_scope(t2)
            assert ss.get_secret("K") == "a"
        finally:
            ss.reset_secret_scope(t1)


class TestEnvFileParsing:
    """load_env_file parses without mutating os.environ."""

    def test_parses_basic(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "# comment\n"
            "ANTHROPIC_API_KEY=sk-abc\n"
            "export OPENAI_API_KEY=sk-def\n"
            'QUOTED="quoted-value"\n'
            "SINGLE='single'\n"
            "\n"
            "BAD_LINE_NO_EQUALS\n"
        )
        out = ss.load_env_file(env)
        assert out == {
            "ANTHROPIC_API_KEY": "sk-abc",
            "OPENAI_API_KEY": "sk-def",
            "QUOTED": "quoted-value",
            "SINGLE": "single",
        }

    def test_does_not_mutate_environ(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ZZZ_KEY", raising=False)
        env = tmp_path / ".env"
        env.write_text("ZZZ_KEY=secret\n")
        ss.load_env_file(env)
        import os
        assert "ZZZ_KEY" not in os.environ

    def test_missing_file_returns_empty(self, tmp_path):
        assert ss.load_env_file(tmp_path / "nope.env") == {}

    def test_build_profile_secret_scope(self, tmp_path):
        (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-profile\n")
        assert ss.build_profile_secret_scope(tmp_path) == {
            "ANTHROPIC_API_KEY": "sk-profile"
        }

    def test_named_profile_inherits_root_shared_secrets(self, tmp_path):
        profile_home = tmp_path / "profiles" / "lingjun"
        profile_home.mkdir(parents=True)
        (tmp_path / ".env").write_text(
            "HERMES_PROVIDER_COMPANY_PROXY_API_KEY=shared-provider\n"
        )
        (profile_home / ".env").write_text(
            "TELEGRAM_BOT_TOKEN=lingjun-identity\n"
        )

        assert ss.build_profile_secret_scope(profile_home) == {
            "HERMES_PROVIDER_COMPANY_PROXY_API_KEY": "shared-provider",
            "TELEGRAM_BOT_TOKEN": "lingjun-identity",
        }

    def test_named_profile_layer_overrides_root_baseline(self, tmp_path):
        profile_home = tmp_path / "profiles" / "lingjun"
        profile_home.mkdir(parents=True)
        (tmp_path / ".env").write_text("TOKEN=root-value\n")
        (profile_home / ".env").write_text("TOKEN=profile-value\n")

        assert ss.build_profile_secret_scope(profile_home)["TOKEN"] == (
            "profile-value"
        )

    def test_named_profile_scopes_do_not_mix_identity_secrets(self, tmp_path):
        lingjun_home = tmp_path / "profiles" / "lingjun"
        companion_home = tmp_path / "profiles" / "companion"
        lingjun_home.mkdir(parents=True)
        companion_home.mkdir(parents=True)
        (tmp_path / ".env").write_text("SHARED_KEY=shared\n")
        (lingjun_home / ".env").write_text("IDENTITY_KEY=lingjun\n")
        (companion_home / ".env").write_text("IDENTITY_KEY=companion\n")

        lingjun = ss.build_profile_secret_scope(lingjun_home)
        companion = ss.build_profile_secret_scope(companion_home)

        assert lingjun == {"SHARED_KEY": "shared", "IDENTITY_KEY": "lingjun"}
        assert companion == {
            "SHARED_KEY": "shared",
            "IDENTITY_KEY": "companion",
        }
