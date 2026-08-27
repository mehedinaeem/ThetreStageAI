from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("theatre", "0005_generationrun_model_settings_and_more")]

    operations = [
        migrations.AddField(
            model_name="generationrun",
            name="validation_history",
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text="Separate initial and final validation errors for repair research.",
            ),
        ),
    ]
