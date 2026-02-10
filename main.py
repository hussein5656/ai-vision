"""Professional surveillance system launcher."""

import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ui.app_professional import main

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*20 + "SYSTEME DE SURVEILLANCE PROFESSIONNEL v1.0")
    print("="*80)
    print("[OK] Affichage detaille avec boites englobantes et IDs")
    print("[OK] Suivi professionnel des objets en temps reel")
    print("[OK] Systeme d'alertes multi-niveaux")
    print("[OK] Gestion des zones interdites et zones de parking")
    print("[OK] Detection de stationnement prolonge")
    print("[OK] Detection de trainards (loitering)")
    print("[OK] Notification d'anomalies")
    print("[OK] Statistiques completes en temps reel")
    print("[OK] Support multi-sources (cameras, streams, videos, images)")
    print("="*80 + "\n")
    
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)