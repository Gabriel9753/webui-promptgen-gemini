import launch

if not launch.is_installed("google-generativeai"):
    launch.run_pip("install google-generativeai", "requirements for Promptgen")

if not launch.is_installed("loguru"):
    launch.run_pip("install loguru", "requirements for Promptgen")
