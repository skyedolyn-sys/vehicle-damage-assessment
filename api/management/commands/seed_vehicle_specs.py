"""Django management command to seed vehicle specs cache."""

from django.core.management.base import BaseCommand

from data.vehicle_specs_cache import save_cached_specs
from scripts.seed_vehicle_specs_cache import COMMON_CHINESE_VEHICLES


class Command(BaseCommand):
    help = "Seed vehicle specs cache with common Chinese-market vehicles"

    def handle(self, *args, **options):
        count = 0
        for entry in COMMON_CHINESE_VEHICLES:
            vehicle_info = entry["vehicle_info"]
            specs = entry["specs"]
            save_cached_specs(vehicle_info, specs)
            count += 1

        self.stdout.write(
            self.style.SUCCESS(f"Successfully seeded {count} vehicle entries into cache.")
        )
