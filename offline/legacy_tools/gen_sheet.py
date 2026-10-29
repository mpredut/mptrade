"""A historical report generator; it is not part of the runtime."""

import pandas as pd

# A function that generates the timesheet.
def genereaza_fisa_pontaj(month, year):

    month_days = pd.date_range(start=f'{year}-{month}-01', end=f'{year}-{month}-{pd.Timestamp(f"{year}-{month}-01").days_in_month}')

   

    # The timesheet template, adapted to your model.

    pontaj_data = {

        'Data': month_days.strftime('%d.%m.%Y'),

        'Ziua': month_days.strftime('%A'),

        'Start time': ['09:00' for _ in range(len(month_days))],  # You can adjust these values.

        'End time': ['17:00' for _ in range(len(month_days))],  # You can adjust these values.

        'Ore lucrate': [8 for _ in range(len(month_days))],  # Poti ajusta numarul de ore

        'Observatii': ['' for _ in range(len(month_days))]

    }

   

    df = pd.DataFrame(pontaj_data)

   

    # Save the Excel file under the given name.

    file_name = f'fisa_pontaj_{month}_{year}.xlsx'

    df.to_excel(file_name, index=False)

    print(f'The timesheet for {month}/{year} was generated: {file_name}')

 

# Exemplo de utilizare
if __name__ == "__main__":
    genereaza_fisa_pontaj(9, 2024)

