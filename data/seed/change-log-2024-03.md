# Ledger service change log, March 2024

On 2024-03-02 at 22:14 UTC the payments platform deployed ledger-service v2.4.1.
The change reduced the payment retry count from 5 to 1 and cut the database connection pool from 40 to 8.
The deploy was recorded by release engineer Mara Chen. No other production service was deployed that night.
Rollback instructions say to restore ledger-service v2.3.9.
This change log does not state a revenue figure for any incident.
